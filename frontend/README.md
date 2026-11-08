# Frontend

The dashboard interface lives in [web.html](web.html). It contains the
Codespaces-inspired dark theme, a light/dark toggle, navigation, charts, and
training controls. The selected theme is saved in the browser.

The Flask application serves this page, its assets, and the training API from a
single origin. Start the complete application from the repository root with:

```bash
cmd.exe /c start_dashboard.bat
```

On Windows, open `http://127.0.0.1:5000/`. On Codespaces, forward port 5000
through the VS Code Ports tab. Frontend API requests use the same origin, so
there is no separate static server or CORS port configuration.

Static files such as images belong in [assets/](assets/) and are available at
`/assets/<filename>`.
