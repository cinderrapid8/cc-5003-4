"""pytest bootstrap: make the user-site numpy importable under `python -I`.

The service itself only depends on numpy; tests run with the isolated
interpreter, so we add the user site-packages directory explicitly before
any test module imports numpy.
"""
import os
import site
import sys

_user_site = site.getusersitepackages()
if os.path.isdir(_user_site) and _user_site not in sys.path:
    sys.path.insert(0, _user_site)

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
