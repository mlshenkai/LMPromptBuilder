# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/5/16 10:26 AM
# @File: __init__.py
# @Email: mlshenkai@163.com
import pyrootutils

pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)
