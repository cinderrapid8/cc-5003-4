"use strict";

const $ = (id) => document.getElementById(id);
const state = { tasks: new Map(), selected: null, sse: null };

function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2600);
}

function fmt(v, digits = 2) {
  return v === null || v === undefined ? "—" : Number(v).toFixed(digits);
}

function pill(status) {
  return `<span class="pill ${status}">${status}</span>`;
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw Object.assign(new Error(body.error || res.statusText), { body, status: res.status });
  return body;
}

function fileUrl(taskId, rel) {
  return `/api/tasks/${taskId}/file?path=${encodeURIComponent(rel)}`;
}

async function loadVersion() {
  try {
    const v = await api("/api/version");
    $("prov").textContent =
      `service ${v.service_version} · python ${v.python} · numpy ${v.numpy} · ` +
      `${v.components.loudness.algorithm.split(",")[0]} · 真峰值 ${v.components.loudness.tp_oversample}x 过采样`;
  } catch (err) {
    $("prov").textContent = "版本信息不可用";
  }
}

async function upload() {
  const files = $("files").files;
  if (!files.length) return toast("请先选择 WAV 文件");
  $("uploadBtn").disabled = true;
  for (const file of files) {
    const fd = new FormData();
    fd.append("file", file, file.name);
    try {
      const r = await fetch("/api/tasks", { method: "POST", body: fd });
      const body = await r.json();
      if (!r.ok) {
        toast(`${file.name} 被拒绝：${body.error}`);
        if (body.task_id) await refreshTasks();
        continue;
      }
      toast(`${file.name} 分析完成`);
      await refreshTasks();
      subscribe(body.task_id);
    } catch (err) {
      toast(`${file.name} 上传失败：${err.message}`);
    }
  }
  $("uploadBtn").disabled = false;
}

async function refreshTasks() {
  const { tasks } = await api("/api/tasks");
  state.tasks = new Map(tasks.map((t) => [t.task_id, t]));
  renderTable();
  renderTargetSelect();
  renderDetail();
  renderDownloads();
}

function renderTable() {
  const tbody = $("taskTable").querySelector("tbody");
  tbody.innerHTML = "";
  for (const t of state.tasks.values()) {
    const m = t.analysis?.measurement;
    const warns = m?.warnings?.length ? `<span class="warn">${m.warnings.join(", ")}</span>` : (t.error ? `<span class="warn">${t.error}</span>` : "");
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><b>${t.upload.original_filename}</b><br><span class="muted">${t.task_id.slice(0, 8)} · ${(t.upload.size_bytes / 1024).toFixed(1)} KB</span></td>
      <td>${pill(t.status)}</td>
      <td>${m ? fmt(m.duration_s, 3) + " s" : "—"}</td>
      <td>${m ? `${m.channel_layout} (${m.n_channels}ch)` : "—"}</td>
      <td>${fmt(m?.integrated_lufs)}</td>
      <td>${fmt(m?.lra_lu)}</td>
      <td>${fmt(m?.sample_peak_dbfs)}</td>
      <td>${fmt(m?.true_peak_dbtp)}</td>
      <td>${fmt(m?.rms_dbfs)}</td>
      <td class="muted">${m ? `${m.n_gated_blocks}/${m.n_total_blocks}` : "—"}</td>
      <td>${warns || "—"}</td>
      <td><button class="secondary" data-id="${t.task_id}">选择</button></td>`;
    tr.querySelector("button").onclick = () => selectTask(t.task_id);
    tbody.appendChild(tr);
  }
}

function renderTargetSelect() {
  const sel = $("targetTask");
  const keep = sel.value;
  sel.innerHTML = "";
  for (const t of state.tasks.values()) {
    const opt = document.createElement("option");
    opt.value = t.task_id;
    opt.textContent = `${t.upload.original_filename} (${t.task_id.slice(0, 8)}) [${t.status}]`;
    sel.appendChild(opt);
  }
  if (keep && state.tasks.has(keep)) sel.value = keep;
}

function selectTask(taskId) {
  state.selected = taskId;
  $("targetTask").value = taskId;
  renderDetail();
}

async function submitJob() {
  const taskId = $("targetTask").value;
  if (!taskId) return toast("请选择任务");
  const target = {
    integrated_lufs: parseFloat($("tLufs").value),
    true_peak_dbtp: parseFloat($("tTp").value),
    sample_rate: parseInt($("tRate").value, 10),
    layout: $("tLayout").value,
    bit_depth: parseInt($("tBits").value, 10),
  };
  try {
    await api(`/api/tasks/${taskId}/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(target),
    });
    toast("处理已开始");
    subscribe(taskId);
    setTimeout(refreshTasks, 400);
  } catch (err) {
    toast(err.message);
  }
}

async function cancelTask() {
  const taskId = $("targetTask").value;
  if (!taskId) return;
  const r = await api(`/api/tasks/${taskId}/cancel`, { method: "POST" });
  toast(r.cancelled ? "已请求取消，部分产物将被清理" : "任务已结束，无法取消");
  setTimeout(refreshTasks, 500);
}

async function deleteTask() {
  const taskId = state.selected || $("targetTask").value;
  if (!taskId) return toast("请先选择任务");
  if (!confirm(`删除任务 ${taskId.slice(0, 8)} 的全部文件？`)) return;
  const r = await fetch(`/api/tasks/${taskId}`, { method: "DELETE" });
  if (r.ok) {
    toast("任务及文件已清理");
    state.selected = null;
    await refreshTasks();
  }
}

function renderDetail() {
  const task = state.selected ? state.tasks.get(state.selected) : null;
  const ab = $("abArea");
  const measure = $("measureArea");
  if (!task) {
    ab.className = "muted";
    ab.textContent = "选择一个已完成任务后在此对照试听。";
    measure.className = "muted";
    measure.textContent = "尚无处理结果。";
    return;
  }
  const inputRel = task.upload.stored_as;
  const jobs = task.jobs || [];
  ab.classList.remove("muted");
  ab.innerHTML = `
    <div class="grid2">
      <div><b>原音</b><br><audio controls preload="none" src="${fileUrl(task.task_id, inputRel)}"></audio></div>
      <div><b>处理音</b> ${jobs.length ? "" : '<span class="muted">（尚未处理）</span>'}<br>
        ${jobs.map((j) => j.result_path_safe ? "" : "").join("")}
        ${jobs.filter((j) => j.status === "succeeded" && j.chain?.output_path)
          .map((j) => `<audio controls preload="none" src="${fileUrl(task.task_id, j.chain.output_path)}"></audio>
                       <div class="muted">job ${j.job_id} · ${j.status}</div>`)
          .join("") || '<span class="muted">无成功导出</span>'}
      </div>
    </div>`;

  if (!jobs.length) {
    measure.className = "muted";
    measure.textContent = "该任务还没有处理作业。";
    return;
  }
  measure.classList.remove("muted");
  measure.innerHTML = jobs.map((j) => {
    const c = j.chain;
    if (!c) return `<div class="unmet">作业 ${j.job_id}：${j.status} ${j.error ? "— " + j.error : ""}</div>`;
    const s = c.source_measurement, o = c.output_measurement;
    const rows = [
      ["综合响度 (LUFS)", s?.integrated_lufs, o?.integrated_lufs, c.target.integrated_lufs],
      ["真峰值 (dBTP)", s?.true_peak_dbtp, o?.true_peak_dbtp, c.target.true_peak_dbtp],
      ["样本峰值 (dBFS)", s?.sample_peak_dbfs, o?.sample_peak_dbfs, "—"],
      ["RMS (dBFS)", s?.rms_dbfs, o?.rms_dbfs, "—"],
      ["LRA (LU)", s?.lra_lu, o?.lra_lu, "—"],
    ].map(([k, a, b, tg]) =>
      `<tr><td>${k}</td><td>${fmt(a)}</td><td>${fmt(b)}</td><td>${tg === "—" ? "—" : fmt(tg)}</td></tr>`).join("");
    return `
      <details open>
        <summary>作业 ${j.job_id} · ${pill(j.status)}（静态增益 ${fmt(c.static_gain_db)} dB${
          c.limiter ? ` · 限制器最大衰减 ${fmt(c.limiter.max_gain_reduction_db)} dB` : ""
        }）</summary>
        <table style="margin:8px 0;">
          <thead><tr><th>指标</th><th>原音测量</th><th>导出后重测</th><th>目标</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        ${c.unmet_conditions?.length ? `<div class="unmet"><b>仍未满足的条件：</b><ul style="margin:6px 0 0;padding-left:18px;">
          ${c.unmet_conditions.map((u) => `<li>${u}</li>`).join("")}</ul></div>` : ""}
        <div class="chain">${JSON.stringify({ target: c.target, layout_decision: c.layout_decision,
          resampling: c.resampling, static_gain_db: c.static_gain_db, limiter: c.limiter,
          quantisation: c.quantisation, channel_balance: c.channel_balance }, null, 2)}</div>
      </details>`;
  }).join("");
}

async function renderDownloads() {
  const { downloads } = await api("/api/downloads");
  const tbody = $("dlTable").querySelector("tbody");
  tbody.innerHTML = downloads.length ? "" : '<tr><td colspan="4" class="muted">暂无可下载产物</td></tr>';
  for (const d of downloads) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${d.task_id.slice(0, 8)}</td><td>${d.job_id}</td><td>${d.filename}</td>
      <td><a href="${fileUrl(d.task_id, d.output_path)}"><button class="secondary">下载</button></a></td>`;
    tbody.appendChild(tr);
  }
}

function subscribe(taskId) {
  if (state.sse) state.sse.close();
  const es = new EventSource(`/api/tasks/${taskId}/events`);
  es.addEventListener("status", () => refreshTasks());
  es.onerror = () => es.close();
  state.sse = es;
}

$("uploadBtn").onclick = upload;
$("processBtn").onclick = submitJob;
$("cancelBtn").onclick = cancelTask;
$("deleteBtn").onclick = deleteTask;
$("refreshDl").onclick = renderDownloads;
$("targetTask").onchange = (e) => selectTask(e.target.value);

loadVersion();
refreshTasks();
