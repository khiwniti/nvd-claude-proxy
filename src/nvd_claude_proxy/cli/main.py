"""ncp — NVIDIA Claude Proxy CLI.

Primary usage:

    ncp                         # start proxy + launch claude
    ncp proxy                   # start proxy only (no claude)
    ncp models list             # show configured model aliases
    ncp models show ALIAS       # detail for one alias
    ncp config                  # show resolved settings
    ncp test [PROMPT]           # send a live test message to the proxy
    ncp status                  # check whether a proxy is running on the port
    ncp init                    # interactively create a .env file
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .._version import __version__

app = typer.Typer(
    name="ncp",
    help="NVIDIA Claude Proxy — run Claude Code on NVIDIA NIM.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)
models_app = typer.Typer(help="Manage model aliases.", add_completion=False)
app.add_typer(models_app, name="models")

console = Console()
err_console = Console(stderr=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _health_url(host: str, port: int) -> str:
    return f"{_base_url(host, port)}/healthz"


def _wait_for_proxy(host: str, port: int, timeout: float = 20.0) -> bool:
    """Poll /healthz until the proxy is accepting connections."""
    url = _health_url(host, port)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3)
    return False


def _load_settings():
    """Return settings, prompting for NVIDIA_API_KEY if it is missing."""
    from ..config.settings import Settings

    # Fast path — everything already configured.
    try:
        s = Settings()  # type: ignore[call-arg]
        # Propagate into os.environ so child proxy subprocess inherits it.
        os.environ["NVIDIA_API_KEY"] = s.nvidia_api_key
        return s
    except Exception:
        pass

    # Key is missing — prompt the user and offer to save it.
    console.print(
        Panel(
            "Get a [bold]free[/bold] key (no credit card) at "
            "[cyan]https://build.nvidia.com[/cyan]",
            title="[yellow]NVIDIA_API_KEY not set[/yellow]",
            border_style="yellow",
        )
    )
    api_key = typer.prompt("NVIDIA_API_KEY (paste here)", hide_input=False).strip()
    if not api_key:
        err_console.print("[red]No API key provided. Aborting.[/red]")
        raise typer.Exit(1)

    os.environ["NVIDIA_API_KEY"] = api_key

    save = typer.confirm(
        "Save key to ~/.config/nvd-claude-proxy/.env for future runs?",
        default=True,
    )
    if save:
        _save_api_key(api_key)

    try:
        s = Settings()  # type: ignore[call-arg]
        # Ensure all key settings are visible to child processes.
        os.environ["NVIDIA_API_KEY"] = s.nvidia_api_key
        return s
    except Exception as exc:
        err_console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1) from exc


def _save_api_key(api_key: str) -> None:
    env_path = Path.home() / ".config" / "nvd-claude-proxy" / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge with existing file if present.
    lines: list[str] = []
    if env_path.exists():
        lines = [l for l in env_path.read_text().splitlines() if not l.startswith("NVIDIA_API_KEY=")]
    lines.insert(0, f"NVIDIA_API_KEY={api_key}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]✓[/green] Key saved to [dim]{env_path}[/dim]")


def _load_registry(settings=None):
    import warnings
    from ..config.models import load_model_registry
    path = settings.model_config_path if settings else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return load_model_registry(path)


def _startup_banner(host: str, port: int, registry) -> None:
    base = _base_url(host, port)

    # Model table
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    tbl.add_column("alias", style="cyan")
    tbl.add_column("→", style="dim")
    tbl.add_column("nim_id", style="white")
    tbl.add_column("caps", style="dim")

    for alias, spec in list(registry.specs.items())[:8]:
        caps = []
        if spec.supports_tools:
            caps.append("tools")
        if spec.supports_vision:
            caps.append("vision")
        if spec.supports_reasoning:
            caps.append("think")
        tbl.add_row(alias, "→", spec.nvidia_id, " ".join(caps))

    # Export hint
    hint = (
        f"[dim]export[/dim] ANTHROPIC_BASE_URL={base}\n"
        f"[dim]export[/dim] ANTHROPIC_API_KEY=not-used\n"
        f"[dim]export[/dim] ANTHROPIC_MODEL={registry.default_big}"
    )

    console.print()
    console.print(
        Panel.fit(
            tbl,
            title=f"[bold green]nvd-claude-proxy[/bold green] [dim]v{__version__}[/dim]",
            subtitle=f"[green]{base}[/green]",
            border_style="green",
        )
    )
    console.print(Panel(hint, title="[dim]Shell exports[/dim]", border_style="dim"))
    console.print()


# ── ncp  (default: start proxy + claude) ──────────────────────────────────────

@app.callback()
def _root(ctx: typer.Context) -> None:
    """NVIDIA Claude Proxy — run Claude Code on NVIDIA NIM."""


@app.command()
def code(
    model: str = typer.Option(
        None,
        "--model", "-m",
        help="Claude model alias to pass as ANTHROPIC_MODEL.",
    ),
    port: int = typer.Option(None, "--port", "-p", help="Proxy port (overrides PROXY_PORT)."),
    host: str = typer.Option(None, "--host", help="Bind host (overrides PROXY_HOST)."),
    no_claude: bool = typer.Option(False, "--no-claude", help="Start proxy only, don't launch claude."),
    claude_args: str = typer.Option("", "--claude-args", help='Extra args passed to claude, e.g. "--dangerously-skip-permissions".'),
    api_key: str = typer.Option(None, "--api-key", "-k", help="NVIDIA API key (nvapi-…). Overrides NVIDIA_API_KEY env var."),
) -> None:
    """Start the proxy then launch [bold]claude[/bold] automatically.

    Equivalent to [cyan]ncp proxy[/cyan] + waiting for ready + [cyan]claude[/cyan].
    """
    if api_key:
        os.environ["NVIDIA_API_KEY"] = api_key
    settings = _load_settings()
    effective_host = host or settings.proxy_host
    effective_port = port or settings.proxy_port
    registry = _load_registry(settings)
    effective_model = model or registry.default_big

    _run_proxy_and_claude(
        settings=settings,
        host=effective_host,
        port=effective_port,
        model=effective_model,
        launch_claude=not no_claude,
        claude_extra_args=claude_args.split() if claude_args.strip() else [],
    )


def _run_proxy_and_claude(
    *,
    settings,
    host: str,
    port: int,
    model: str,
    launch_claude: bool,
    claude_extra_args: list[str],
) -> None:
    registry = _load_registry(settings)

    # ── reuse existing proxy if already healthy ─────────────────────────────
    if _wait_for_proxy(host, port, timeout=1.0):
        console.print(f"[dim]Reusing proxy already running on {_base_url(host, port)}[/dim]")
        proxy_proc = None

        def _stop_proxy():
            pass  # not ours to stop
    else:
        # ── start the proxy as a subprocess ───────────────────────────────────
        env = os.environ.copy()
        env["PROXY_HOST"] = host
        env["PROXY_PORT"] = str(port)

        proxy_proc = subprocess.Popen(
            [sys.executable, "-m", "nvd_claude_proxy.main"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        def _stop_proxy():
            if proxy_proc and proxy_proc.poll() is None:
                proxy_proc.terminate()
                try:
                    proxy_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proxy_proc.kill()

        # ── wait for readiness ─────────────────────────────────────────────────
        with console.status("[bold green]Starting proxy…[/bold green]"):
            ready = _wait_for_proxy(host, port, timeout=20.0)

        if not ready:
            err_console.print("[red]Proxy did not start in time. Check NVIDIA_API_KEY.[/red]")
            _stop_proxy()
            raise typer.Exit(1)

    _startup_banner(host, port, registry)

    if not launch_claude:
        console.print("[green]Proxy running.[/green]  Press [bold]Ctrl+C[/bold] to stop.\n")
        try:
            if proxy_proc:
                proxy_proc.wait()
            else:
                signal.pause()  # wait indefinitely if we reused an existing proxy
        except (KeyboardInterrupt, AttributeError):
            console.print("\n[dim]Stopping proxy…[/dim]")
        finally:
            _stop_proxy()
        return

    # ── launch claude ──────────────────────────────────────────────────────────
    claude_bin = _find_claude()
    if claude_bin is None:
        err_console.print(
            "[yellow]Warning:[/yellow] [bold]claude[/bold] not found in PATH. "
            "Install it with [cyan]npm install -g @anthropic-ai/claude-code[/cyan]\n"
            "The proxy is running — connect any Anthropic-compatible client."
        )
        console.print("[green]Proxy running.[/green]  Press [bold]Ctrl+C[/bold] to stop.\n")
        try:
            proxy_proc.wait()
        except KeyboardInterrupt:
            pass
        finally:
            _stop_proxy()
        return

    claude_env = os.environ.copy()
    claude_env["ANTHROPIC_BASE_URL"] = _base_url(host, port)
    claude_env["ANTHROPIC_API_KEY"] = claude_env.get("ANTHROPIC_API_KEY") or "ncp-local"
    claude_env["ANTHROPIC_MODEL"] = model
    claude_env["ANTHROPIC_SMALL_FAST_MODEL"] = registry.default_small
    # Set output token limit to the model's max_output so Claude Code never
    # hits "response exceeded maximum" errors on long tool outputs.
    primary_spec = registry.resolve(model)
    claude_env.setdefault(
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS", str(primary_spec.max_output)
    )

    claude_cmd = [claude_bin, *claude_extra_args]
    console.print(f"[dim]Launching:[/dim] [cyan]{' '.join(claude_cmd)}[/cyan]\n")

    try:
        result = subprocess.run(claude_cmd, env=claude_env)
    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n[dim]Stopping proxy…[/dim]")
        _stop_proxy()

    raise typer.Exit(getattr(result, "returncode", 0))


def _find_claude() -> str | None:
    """Return the full path to the `claude` binary, or None if not found."""
    import shutil
    return shutil.which("claude")


# ── ncp proxy ─────────────────────────────────────────────────────────────────

@app.command()
def proxy(
    port: int = typer.Option(None, "--port", "-p"),
    host: str = typer.Option(None, "--host"),
    api_key: str = typer.Option(None, "--api-key", "-k", help="NVIDIA API key (nvapi-…). Overrides NVIDIA_API_KEY env var."),
) -> None:
    """Start the proxy server only (no [bold]claude[/bold])."""
    if api_key:
        os.environ["NVIDIA_API_KEY"] = api_key
    settings = _load_settings()
    effective_host = host or settings.proxy_host
    effective_port = port or settings.proxy_port
    _run_proxy_and_claude(
        settings=settings,
        host=effective_host,
        port=effective_port,
        model="",
        launch_claude=False,
        claude_extra_args=[],
    )


# ── ncp status ────────────────────────────────────────────────────────────────

@app.command()
def status(
    port: int = typer.Option(None, "--port", "-p"),
    host: str = typer.Option(None, "--host"),
) -> None:
    """Check whether the proxy is running and print model info."""
    settings = _load_settings()
    effective_host = host or settings.proxy_host
    effective_port = port or settings.proxy_port
    base = _base_url(effective_host, effective_port)

    try:
        r = httpx.get(f"{base}/healthz", timeout=3.0)
        if r.status_code == 200:
            console.print(f"[green]● Proxy is UP[/green]  {base}")
            # Also hit /v1/models
            mr = httpx.get(f"{base}/v1/models", timeout=3.0)
            if mr.status_code == 200:
                models = mr.json().get("data", [])
                console.print(f"  Models served: {', '.join(m['id'] for m in models[:5])}")
        else:
            console.print(f"[yellow]● Unexpected status {r.status_code}[/yellow]  {base}")
    except Exception:
        console.print(f"[red]● Proxy is DOWN[/red]  {base}")
        console.print(f"  Run [cyan]ncp proxy[/cyan] or [cyan]ncp[/cyan] to start it.")
        raise typer.Exit(1)


# ── ncp models list ───────────────────────────────────────────────────────────

@models_app.command("list")
def models_list() -> None:
    """List all configured model aliases."""
    settings = _load_settings()
    registry = _load_registry(settings)

    tbl = Table(
        title="Model aliases",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
    )
    tbl.add_column("Alias", style="cyan", no_wrap=True)
    tbl.add_column("NVIDIA NIM id", style="white")
    tbl.add_column("Tools", justify="center")
    tbl.add_column("Vision", justify="center")
    tbl.add_column("Reasoning", justify="center")
    tbl.add_column("Failover →", style="dim")

    for alias, spec in registry.specs.items():
        marker = lambda v: "[green]✓[/green]" if v else "[dim]–[/dim]"  # noqa: E731
        default_tag = ""
        if alias == registry.default_big:
            default_tag = " [dim](big)[/dim]"
        elif alias == registry.default_small:
            default_tag = " [dim](small)[/dim]"
        tbl.add_row(
            alias + default_tag,
            spec.nvidia_id,
            marker(spec.supports_tools),
            marker(spec.supports_vision),
            marker(spec.supports_reasoning),
            ", ".join(spec.failover_to) or "–",
        )

    console.print(tbl)


# ── ncp models show ───────────────────────────────────────────────────────────

@models_app.command("show")
def models_show(alias: str = typer.Argument(..., help="Model alias to inspect.")) -> None:
    """Show full configuration for a single model alias."""
    settings = _load_settings()
    registry = _load_registry(settings)

    spec = registry.specs.get(alias)
    if spec is None:
        # Try prefix fallback
        resolved = registry.resolve(alias)
        console.print(
            f"[yellow]'{alias}' not found directly; resolved to '{resolved.alias}'[/yellow]"
        )
        spec = resolved

    rows = [
        ("Alias", spec.alias),
        ("NVIDIA NIM id", spec.nvidia_id),
        ("Supports tools", str(spec.supports_tools)),
        ("Supports vision", str(spec.supports_vision)),
        ("Supports reasoning", str(spec.supports_reasoning)),
        ("Reasoning style", spec.reasoning_style),
        ("Max context tokens", f"{spec.max_context:,}"),
        ("Max output tokens", f"{spec.max_output:,}"),
        ("Temperature override", str(spec.temperature_override) if spec.temperature_override else "–"),
        ("Failover chain", ", ".join(spec.failover_to) if spec.failover_to else "–"),
    ]

    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    tbl.add_column("field", style="dim")
    tbl.add_column("value", style="white")
    for k, v in rows:
        tbl.add_row(k, v)

    console.print(
        Panel(tbl, title=f"[bold cyan]{spec.alias}[/bold cyan]", border_style="cyan")
    )


# ── ncp config ────────────────────────────────────────────────────────────────

@app.command()
def config() -> None:
    """Show resolved proxy configuration (API key is masked)."""
    settings = _load_settings()

    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    tbl.add_column("var", style="dim")
    tbl.add_column("value", style="white")

    api_key = settings.nvidia_api_key
    masked = api_key[:8] + "…" + api_key[-4:] if len(api_key) > 12 else "***"

    rows = [
        ("NVIDIA_API_KEY", masked),
        ("NVIDIA_BASE_URL", settings.nvidia_base_url),
        ("PROXY_HOST", settings.proxy_host),
        ("PROXY_PORT", str(settings.proxy_port)),
        ("PROXY_API_KEY", "set" if settings.proxy_api_key else "unset"),
        ("LOG_LEVEL", settings.log_level),
        ("MODEL_CONFIG_PATH", settings.model_config_path),
        ("REQUEST_TIMEOUT_SECONDS", str(settings.request_timeout_seconds)),
        ("MAX_RETRIES", str(settings.max_retries)),
        ("RATE_LIMIT_RPM", str(settings.rate_limit_rpm) + (" (disabled)" if settings.rate_limit_rpm == 0 else "")),
        ("MAX_REQUEST_BODY_MB", str(settings.max_request_body_mb) + (" (disabled)" if settings.max_request_body_mb == 0 else "")),
    ]
    for k, v in rows:
        tbl.add_row(k, v)

    console.print(Panel(tbl, title="[bold]nvd-claude-proxy config[/bold]", border_style="dim"))


# ── ncp test ──────────────────────────────────────────────────────────────────

@app.command()
def test(
    prompt: str = typer.Argument("Say 'proxy OK' in exactly 3 words."),
    model: str = typer.Option(None, "--model", "-m"),
    port: int = typer.Option(None, "--port", "-p"),
    host: str = typer.Option(None, "--host"),
) -> None:
    """Send a test message to the running proxy and print the response."""
    settings = _load_settings()
    effective_host = host or settings.proxy_host
    effective_port = port or settings.proxy_port
    base = _base_url(effective_host, effective_port)

    registry = _load_registry(settings)
    effective_model = model or registry.default_big

    payload = {
        "model": effective_model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": prompt}],
    }

    headers = {"content-type": "application/json"}
    if settings.proxy_api_key:
        headers["x-api-key"] = settings.proxy_api_key

    console.print(f"[dim]POST {base}/v1/messages  model={effective_model}[/dim]")
    try:
        with console.status("Waiting for response…"):
            r = httpx.post(
                f"{base}/v1/messages",
                json=payload,
                headers=headers,
                timeout=30.0,
            )
    except httpx.ConnectError:
        err_console.print(f"[red]Cannot connect to {base}[/red] — is the proxy running?")
        raise typer.Exit(1)

    if r.status_code != 200:
        err_console.print(f"[red]HTTP {r.status_code}[/red]")
        console.print_json(r.text)
        raise typer.Exit(1)

    body = r.json()
    text = ""
    for block in body.get("content", []):
        if block.get("type") == "text":
            text = block["text"]
            break

    usage = body.get("usage", {})
    console.print(
        Panel(
            text or "[dim](no text content)[/dim]",
            title=f"[green]✓ HTTP 200[/green]  stop=[bold]{body.get('stop_reason', '?')}[/bold]"
                  f"  in={usage.get('input_tokens','?')} out={usage.get('output_tokens','?')}",
            border_style="green",
        )
    )


# ── ncp init ──────────────────────────────────────────────────────────────────

@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing .env."),
) -> None:
    """Interactively create a [bold].env[/bold] file in the current directory."""
    # Prefer cwd/.env for local/dev use; fall back to global XDG location.
    local_env = Path(".env")
    global_env = Path.home() / ".config" / "nvd-claude-proxy" / ".env"
    env_path = local_env if local_env.exists() else global_env

    if env_path.exists() and not force:
        overwrite = typer.confirm(f"{env_path} already exists. Overwrite?", default=False)
        if not overwrite:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit()

    console.print(
        Panel(
            "Get a [bold]free[/bold] NVIDIA API key at [cyan]https://build.nvidia.com[/cyan]\n"
            "(no credit card required)",
            title="NVIDIA API key",
            border_style="cyan",
        )
    )
    api_key = typer.prompt("NVIDIA_API_KEY (paste here)", hide_input=False).strip()

    port = typer.prompt("Proxy port", default="8788")
    host = typer.prompt("Bind host (127.0.0.1 = local only, 0.0.0.0 = all)", default="127.0.0.1")
    proxy_key = typer.prompt(
        "Proxy API key (leave blank = no auth)", default="", hide_input=True
    )
    log_level = typer.prompt("Log level", default="INFO")

    lines = [
        f"NVIDIA_API_KEY={api_key}",
        f"PROXY_PORT={port}",
        f"PROXY_HOST={host}",
        f"LOG_LEVEL={log_level}",
    ]
    if proxy_key.strip():
        lines.append(f"PROXY_API_KEY={proxy_key}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"\n[green]✓[/green] Written [bold]{env_path}[/bold]")
    console.print(
        "\nStart the proxy:\n"
        f"  [cyan]ncp[/cyan]                  # start proxy + claude\n"
        f"  [cyan]ncp proxy[/cyan]            # proxy only\n"
        f"  [cyan]ncp models list[/cyan]      # show model aliases\n"
    )


# ── ncp kill ──────────────────────────────────────────────────────────────────

@app.command()
def kill(
    port: int = typer.Option(None, "--port", "-p"),
    host: str = typer.Option(None, "--host"),
) -> None:
    """Kill any process listening on the proxy port (cleans up stuck instances)."""
    import signal as _signal

    settings = _load_settings()
    effective_port = port or settings.proxy_port

    try:
        import subprocess as _sp
        # lsof works on macOS/Linux; find PIDs listening on TCP port
        result = _sp.run(
            ["lsof", "-ti", f"tcp:{effective_port}"],
            capture_output=True, text=True,
        )
        pids = [p.strip() for p in result.stdout.splitlines() if p.strip()]
        if not pids:
            console.print(f"[dim]Nothing listening on port {effective_port}.[/dim]")
            return
        for pid in pids:
            try:
                os.kill(int(pid), _signal.SIGTERM)
                console.print(f"[green]✓[/green] Killed PID {pid} (port {effective_port})")
            except ProcessLookupError:
                pass
    except FileNotFoundError:
        err_console.print("[yellow]lsof not found — cannot auto-kill. Run:[/yellow]")
        err_console.print(f"  kill $(lsof -ti tcp:{effective_port})")


# ── version ───────────────────────────────────────────────────────────────────

@app.command()
def version() -> None:
    """Print version and exit."""
    console.print(f"nvd-claude-proxy [cyan]{__version__}[/cyan]")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app()


if __name__ == "__main__":
    main()
