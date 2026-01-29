# Installation

## Local install
```
{
  "mcpServers": {
    "cefi_mcp": {
      "command": "uv", // May need a full path, like /opt/homebrew/bin/uv
      "args": [
        "--directory",
        "/Absolute/path/to/cefi-mcp-pilot",
        "run",
        "cefi-mcp"
      ]
    }
  },
}
```

## Local install through Docker / Podman

Docker or Podman should work interchangeably.

Build and run:
```
docker build -t cefi-mcp-pilot . && docker run -p 8080:8080 cefi-mcp-pilot
```

Then, connect the client to `http://localhost:8080/mcp`

# Tips

The "Geocode ocean place" tool is helpful for getting the coordinates of less commonly known locations. It can be enabled by including the `--use_geocode` flag in the command used to launch the server. However, for common locations, it can return a number of similar but different coordinates. This tool needs additional development work to enable it to identify the best coordinates to return.

# Disclaimers

This is an experimental pilot project powered by generative AI. Information is for demonstration purposes only, may contain errors, and is not a substitute for official NOAA forecasts, alerts, or data.

The United States Department of Commerce (DOC) GitHub project code is provided on an 'as is' basis and the user assumes responsibility for its use. The DOC has relinquished control of the information and no longer has responsibility to protect the integrity, confidentiality, or availability of the information. Any claims against the Department of Commerce stemming from the use of its GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.

This project code is made available through GitHub but is managed by NOAA-GFDL at https://www.gfdl.noaa.gov.
