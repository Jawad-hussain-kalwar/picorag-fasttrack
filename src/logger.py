import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from src.config import LOG_COLOR, LOG_LEVEL

try:
    from colorama import Back, Fore, Style, init

    init(autoreset=True)
except ImportError:  # pragma: no cover
    class _Dummy:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""

    class _DummyStyle:
        BRIGHT = RESET_ALL = ""

    Back = Fore = _Dummy()
    Style = _DummyStyle()


class Ansi:
    ITALIC = "\x1b[3m"
    RESET_ITALIC = "\x1b[23m"

    @staticmethod
    def fg_256(code: int) -> str:
        return f"\x1b[38;5;{code}m"

    @staticmethod
    def bg_256(code: int) -> str:
        return f"\x1b[48;5;{code}m"


@dataclass(frozen=True)
class StyleSpec:
    fg: str = ""
    bg: str = ""
    bold: bool = False
    italic: bool = False


def style(fg: str = "", bg: str = "", *, bold: bool = False, italic: bool = False) -> StyleSpec:
    return StyleSpec(fg=fg, bg=bg, bold=bold, italic=italic)


# Central palette for easy manual edits.
COLORS = {
    "dark_blue": Ansi.fg_256(24),
    "dark_brown": Ansi.fg_256(94),
    "dark_magenta": Ansi.fg_256(90),
    "hot_magenta": Ansi.fg_256(201),
    "bg_light_blue": Ansi.bg_256(153),
    "bg_light_warm": Ansi.bg_256(230),
    "bg_warm_yellow": Ansi.bg_256(223),
    "bg_cyan": Ansi.bg_256(159),
    "bg_green_dark": Ansi.bg_256(28),
    "bg_green_bright": Ansi.bg_256(120),
    "bg_white": Ansi.bg_256(255),
    "bg_indigo": Ansi.bg_256(57),
}


def build_level_styles() -> dict[int, StyleSpec]:
    return {
        logging.DEBUG: style(fg=Fore.CYAN),
        logging.INFO: style(fg=Fore.GREEN, bold=True),
        logging.WARNING: style(fg=Fore.YELLOW, bold=True),
        logging.ERROR: style(fg=Fore.RED, bold=True),
        logging.CRITICAL: style(fg=Fore.WHITE, bg=Back.RED, bold=True),
    }


def build_event_styles() -> dict[str, StyleSpec]:
    # Edit this mapping when you want to tune event colors.
    return {
        "startup": style(fg=COLORS["dark_blue"], bg=COLORS["bg_light_blue"], bold=True),
        "shutdown": style(fg=COLORS["dark_blue"], bg=COLORS["bg_light_warm"], bold=True),
        "user_input": style(fg=Fore.BLACK, bg=Back.WHITE, bold=True),
        "info": style(fg=Fore.GREEN, italic=True),
        "debug": style(fg=Fore.CYAN, italic=True),
        "warning": style(fg=COLORS["dark_brown"], bg=COLORS["bg_warm_yellow"], bold=True),
        "index_start": style(fg=Ansi.fg_256(17), bg=COLORS["bg_cyan"], bold=True),
        "index_doc": style(fg=Fore.YELLOW, italic=True),
        "index_done": style(fg=Fore.WHITE, bg=COLORS["bg_green_dark"], bold=True),
        "retrieve_start": style(fg=COLORS["dark_magenta"], bg=COLORS["bg_white"], bold=True),
        "retrieve_done": style(fg=COLORS["hot_magenta"], bg=COLORS["bg_indigo"], bold=True),
        "retrieve_empty": style(fg=Fore.WHITE, bg=Back.RED, bold=True),
        "llm_request": style(fg=Fore.BLACK, bg=Back.CYAN, italic=True),
        "llm_response": style(fg=Fore.BLACK, bg=Back.GREEN, italic=True),
        "success": style(fg=Fore.BLACK, bg=COLORS["bg_green_bright"], bold=True),
        "error": style(fg=Fore.WHITE, bg=Back.RED, bold=True),
    }


LEVEL_STYLES = build_level_styles()
EVENT_STYLES = build_event_styles()
FALLBACK_EVENT_STYLE = style(fg=Fore.WHITE, bg=Back.BLACK)


def _style_text(text: str, spec: StyleSpec, use_color: bool) -> str:
    if not use_color:
        return text

    prefix = f"{spec.fg}{spec.bg}"
    if spec.bold:
        prefix += Style.BRIGHT
    if spec.italic:
        prefix += Ansi.ITALIC

    suffix = f"{Ansi.RESET_ITALIC}{Style.RESET_ALL}" if spec.italic else Style.RESET_ALL
    return f"{prefix}{text}{suffix}"


class FastStyleFormatter(logging.Formatter):
    def __init__(
        self,
        *,
        use_color: bool = True,
        level_styles: dict[int, StyleSpec] | None = None,
        event_styles: dict[str, StyleSpec] | None = None,
        fallback_event_style: StyleSpec | None = None,
    ) -> None:
        super().__init__()
        self.use_color = use_color
        self.level_styles = level_styles or LEVEL_STYLES
        self.event_styles = event_styles or EVENT_STYLES
        self.fallback_event_style = fallback_event_style or FALLBACK_EVENT_STYLE

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname.upper()
        level_spec = self.level_styles.get(record.levelno, StyleSpec())
        level_text = _style_text(level, level_spec, self.use_color)

        event = getattr(record, "event", None)
        event_text = ""
        if event:
            event_spec = self.event_styles.get(event, self.fallback_event_style)
            event_text = f" {_style_text(f'[{event}]', event_spec, self.use_color)}"

        payload = getattr(record, "payload", None)
        payload_text = ""
        if payload:
            parts = [f"{k}={v}" for k, v in payload.items()]
            payload_text = " " + " ".join(parts)

        return f"{level_text}:     {record.getMessage()}{event_text}{payload_text}"


class RAGLogger:
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def event(self, level: int, message: str, *, event: str, **payload: Any) -> None:
        self._logger.log(level, message, extra={"event": event, "payload": payload or None})

    def debug(self, message: str, *, event: str = "debug", **payload: Any) -> None:
        self.event(logging.DEBUG, message, event=event, **payload)

    def info(self, message: str, *, event: str = "info", **payload: Any) -> None:
        self.event(logging.INFO, message, event=event, **payload)

    def warning(self, message: str, *, event: str = "warning", **payload: Any) -> None:
        self.event(logging.WARNING, message, event=event, **payload)

    def error(self, message: str, *, event: str = "error", **payload: Any) -> None:
        self.event(logging.ERROR, message, event=event, **payload)

    def exception(self, message: str, *, event: str = "error", **payload: Any) -> None:
        self._logger.exception(message, extra={"event": event, "payload": payload or None})

    def timer(self, message: str, *, event: str):
        return _Timer(self, message, event=event)


class _Timer:
    def __init__(self, logger: RAGLogger, message: str, *, event: str):
        self.logger = logger
        self.message = message
        self.event = event
        self.start = 0.0

    def __enter__(self) -> "_Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed_ms = int((time.perf_counter() - self.start) * 1000)
        if exc is None:
            self.logger.info(self.message, event=self.event, elapsed_ms=elapsed_ms)
        else:
            self.logger.error(
                f"{self.message} failed",
                event="error",
                elapsed_ms=elapsed_ms,
                reason=str(exc),
            )
        return None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no"}


def get_logger(name: str = "pico-rag") -> RAGLogger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return RAGLogger(logger)

    level_name = os.getenv("LOG_LEVEL", LOG_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)
    use_color = _env_bool("LOG_COLOR", default=LOG_COLOR)

    handler = logging.StreamHandler()
    handler.setFormatter(FastStyleFormatter(use_color=use_color))

    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return RAGLogger(logger)


def _showcase_logs() -> None:
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    os.environ.setdefault("LOG_COLOR", "1")

    log = get_logger("pico-rag-showcase")
    log._logger.info("Logger style showcase (levels, events, payload, exception)")

    log.debug("Debug level sample", step="parse")
    log.info("Info level sample", component="pipeline")
    log.warning("Warning level sample", threshold=0.42)
    log.error("Error level sample", code="E_DEMO")
    log.event(logging.CRITICAL, "Critical level sample", event="error", code="E_CRIT")

    for event_name in EVENT_STYLES:
        log.info(f"Event style sample for '{event_name}'", event=event_name, sample=True)
    log.info("Unknown event fallback style", event="unknown_event", sample=True)

    with log.timer("Timer success sample", event="success"):
        time.sleep(0.02)

    try:
        with log.timer("Timer failure sample", event="index_done"):
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass

    try:
        raise ValueError("simulated exception")
    except ValueError:
        log.exception("Exception sample with traceback", context="showcase")


if __name__ == "__main__":
    _showcase_logs()
