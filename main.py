#!/usr/bin/env python3
"""AILLM-SEC - AI Security Analysis Assistant."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
