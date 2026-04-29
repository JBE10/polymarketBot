"""
macOS Secure Enclave / Keychain key management.

La clave privada se guarda en el Llavero de macOS como una contraseña genérica.
En Macs con Apple Silicon o chip T2, el ítem puede estar protegido por el
Secure Enclave (hardware). Accedemos exclusivamente a través del CLI `security`
del sistema operativo — la clave nunca toca un archivo en disco.

Configuración inicial (ejecutar UNA sola vez desde tu Terminal de IntelliJ):
──────────────────────────────────────────────────────────────────────────────
    security add-generic-password \\
        -s "polymarket_pk" \\
        -a "bot_user" \\
        -w "TU_CLAVE_PRIVADA" \\
        -T "/Applications/IntelliJ IDEA.app"

O usando el helper interactivo (más seguro — no queda en el historial):
    python -c "from src.core.security import store_key; store_key()"
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import getpass
import logging
import os
import subprocess
import sys
from typing import Optional

log = logging.getLogger(__name__)

# Nombres por defecto — coinciden con el comando que ejecutaste en la Terminal
_DEFAULT_SERVICE = "polymarket_pk"
_DEFAULT_ACCOUNT = "bot_user"

# Ruta de IntelliJ IDEA (se añade al ACL para que no pida contraseña cada vez)
_INTELLIJ_PATHS = [
    "/Applications/IntelliJ IDEA.app",
    "/Applications/IntelliJ IDEA CE.app",   # Community Edition
    "/Applications/IntelliJ IDEA Ultimate.app",
]


# ── Lectura ───────────────────────────────────────────────────────────────────

def get_private_key(
    service: str = _DEFAULT_SERVICE,
    account: str = _DEFAULT_ACCOUNT,
) -> str:
    """
    Extrae la clave privada provista vía variable de entorno (para VPS/Docker)
    o del Llavero de macOS en tiempo de ejecución.

    Prioridad: config (PRIVATE_KEY) → Keychain macOS → Lanza PermissionError.
    """
    from src.core.config import get_settings

    cfg_key = get_settings().private_key
    if cfg_key:
        clave = cfg_key.strip()
        bare = clave.removeprefix("0x")
        if len(bare) != 64 or not all(c in "0123456789abcdefABCDEF" for c in bare):
            raise ValueError(
                "La variable PRIVATE_KEY configurada no parece una clave privada "
                "Ethereum válida (se esperan 64 caracteres hexadecimales)."
            )
        log.debug("Clave privada recuperada desde configuración/variable de entorno.")
        return clave

    # Si no hay env var, intentamos Keychain (macOS default)
    comando = [
        "security",
        "find-generic-password",
        "-s", service,
        "-a", account,
        "-w",   # devuelve solo el valor de la contraseña
    ]

    try:
        resultado = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            check=True,         # lanza CalledProcessError si rc != 0
            env=os.environ,
            timeout=10,
        )
        clave = resultado.stdout.strip()

        if not clave:
            raise PermissionError(
                f"El Llavero devolvió una clave vacía para service='{service}', account='{account}'."
            )

        # Validación mínima: clave Ethereum = 64 hex chars (con o sin '0x')
        bare = clave.removeprefix("0x")
        if len(bare) != 64 or not all(c in "0123456789abcdefABCDEF" for c in bare):
            raise ValueError(
                "El valor recuperado no parece una clave privada Ethereum válida "
                "(se esperan 64 caracteres hexadecimales)."
            )

        return clave

    except subprocess.CalledProcessError as exc:
        raise PermissionError(
            "CRÍTICO: No se pudo extraer la clave del Secure Enclave. "
            f"Código de error: {exc.returncode}. "
            "Verifica que ejecutaste 'security add-generic-password' correctamente "
            "y que IntelliJ IDEA tiene permisos en el Llavero de macOS."
        ) from exc

    except subprocess.TimeoutExpired as exc:
        raise PermissionError("Timeout consultando el Llavero de macOS.") from exc

    except FileNotFoundError as exc:
        raise PermissionError(
            "No se encontró el comando `security`. Este módulo requiere macOS."
        ) from exc


def get_private_key_optional(
    service: str = _DEFAULT_SERVICE,
    account: str = _DEFAULT_ACCOUNT,
) -> Optional[str]:
    """
    Versión que retorna None en lugar de lanzar excepción.
    Usada por src/main.py para manejar el error de forma controlada.
    """
    try:
        return get_private_key(service, account)
    except (PermissionError, ValueError) as exc:
        log.warning("Keychain lookup failed: %s", exc)
        return None


# ── Escritura (helper interactivo) ────────────────────────────────────────────

def store_key(
    service: str = _DEFAULT_SERVICE,
    account: str = _DEFAULT_ACCOUNT,
) -> bool:
    """
    Helper interactivo para guardar la clave privada en el Llavero.

    Usa getpass para que la clave NO aparezca en el historial del terminal.
    Idempotente — borra el ítem anterior antes de crear uno nuevo.
    Añade al ACL tanto IntelliJ IDEA como el intérprete Python activo.
    """
    print(
        "\n[Configuración del Secure Enclave]\n"
        "Ingresa tu clave privada Ethereum (hex, con o sin prefijo 0x).\n"
        "El valor NO se mostrará en pantalla y se guardará en el Llavero de macOS.\n"
    )
    clave = getpass.getpass("Clave privada: ").strip()

    bare = clave.removeprefix("0x")
    if len(bare) != 64 or not all(c in "0123456789abcdefABCDEF" for c in bare):
        print("Error: la clave no tiene el formato correcto (64 caracteres hex).")
        return False

    # Borrar ítem existente (ignorar si no existe)
    subprocess.run(
        ["security", "delete-generic-password", "-a", account, "-s", service],
        capture_output=True,
    )

    # Construir la lista de apps de confianza (ACL)
    trusted_apps: list[str] = [sys.executable]   # intérprete Python activo
    for path in _INTELLIJ_PATHS:
        import os as _os
        if _os.path.exists(path):
            trusted_apps.append(path)

    # Armar el comando con múltiples flags -T
    cmd = ["security", "add-generic-password",
           "-a", account, "-s", service, "-w", clave]
    for app in trusted_apps:
        cmd += ["-T", app]
    cmd.append("-U")   # actualizar si ya existe

    resultado = subprocess.run(cmd, capture_output=True, timeout=10)

    if resultado.returncode == 0:
        print(
            f"\n✅ Clave guardada correctamente en el Llavero de macOS.\n"
            f"   Servicio : {service}\n"
            f"   Cuenta   : {account}\n"
            f"   Apps con acceso ACL:\n"
            + "".join(f"     • {a}\n" for a in trusted_apps)
        )
        return True

    print(f"❌ Error al guardar la clave: {resultado.stderr.decode().strip()}")
    return False


def delete_key(
    service: str = _DEFAULT_SERVICE,
    account: str = _DEFAULT_ACCOUNT,
) -> bool:
    """Elimina un ítem del Llavero (útil al rotar claves)."""
    result = subprocess.run(
        ["security", "delete-generic-password", "-a", account, "-s", service],
        capture_output=True,
        timeout=10,
    )
    return result.returncode == 0


# ══════════════════════════════════════════════════════════════════════════════
# TEST INTERMEDIO — Ejecutar con:  python src/core/security.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== TEST DEL MÓDULO DE SEGURIDAD MACOS ===")

    try:
        print(f"Servicio  : {_DEFAULT_SERVICE}")
        print(f"Cuenta    : {_DEFAULT_ACCOUNT}")
        print("Intentando extraer la clave del Keychain...")

        clave_secreta = get_private_key()

        if not clave_secreta:
            print("❌ Error: La clave está vacía.")
        else:
            # Imprimimos solo los primeros y últimos caracteres para no exponerla en consola
            clave_oculta = f"{clave_secreta[:4]}...{clave_secreta[-4:]}"
            print(f"✅ ¡Éxito! Clave recuperada de forma segura en memoria: {clave_oculta}")
            print(f"   Longitud de la clave: {len(clave_secreta)} caracteres.")

    except PermissionError as ex:
        print(f"❌ Fallo de seguridad: {ex}")
        print(
            "\nSolución: Ejecuta primero el siguiente comando en tu Terminal:\n"
            f'  security add-generic-password -s "{_DEFAULT_SERVICE}" '
            f'-a "{_DEFAULT_ACCOUNT}" -w "TU_CLAVE_PRIVADA" '
            '-T "/Applications/IntelliJ IDEA.app"\n'
        )
    except Exception as ex:
        print(f"❌ Error inesperado: {ex}")

    print("==========================================")
