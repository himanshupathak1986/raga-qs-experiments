import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_patch_metrics import patch_metrics


def test_patch_metrics_basic():
    patch = """diff --git a/pkg/a.py b/pkg/a.py
index 111..222 100644
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -1,4 +1,6 @@
 import os
+import subprocess
 def f(x):
-    return x
+    if x:
+        return eval(x)
+    return None
"""
    m = patch_metrics(patch, [r"\beval\s*\(", r"\bsubprocess\."])
    assert m["files_changed"] == 1
    assert m["python_files_changed"] == 1
    assert m["additions"] == 4
    assert m["deletions"] == 1
    assert m["hunks"] == 1
    assert m["imports_added"] == 1
    assert m["control_tokens_added"] == 1
    assert m["security_sensitive_api_additions"] >= 1
