# Frontend

The dashboard interface lives in [web.html](web.html). The local launcher serves
this folder on `http://127.0.0.1:8000/` and maps the root URL to `web.html`.
Static files such as images belong in [assets/](assets/).

The page calls the Flask API at `http://127.0.0.1:5000`. Start both services
together from the repository root with:

```bash
cmd.exe /c start_dashboard.bat
```

The API only allows requests from the local frontend origins configured in
`src/config.py`. When changing either service's port, update both that
configuration and the frontend API URL.
