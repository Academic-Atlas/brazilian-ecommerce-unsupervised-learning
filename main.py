
import os
import sys
import subprocess
import venv

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, ".venv")
REQUIREMENTS = os.path.join(ROOT, "requirements.txt")

PYTHON = os.path.join(
    VENV,
    "Scripts" if os.name == "nt" else "bin",
    "python.exe" if os.name == "nt" else "python"
)

def main():
    os.chdir(ROOT)

    if not os.path.isfile(PYTHON):
        print("CRIANDO AMBIENTE VIRTUAL...", flush=True)
        venv.create(VENV, with_pip=True)

    if sys.prefix != os.path.realpath(VENV):
        subprocess.run(
            [PYTHON, "-m", "pip", "install", "--upgrade", "pip"],
            check=True
        )

        subprocess.run(
            [PYTHON, "-m", "pip", "install", "-r", REQUIREMENTS],
            check=True
        )

        subprocess.run([PYTHON, __file__], check=True)
        return

    etapas = [
        "integrar_dados",
        "tratar_dados",
        "analise_exploratoria",
        "preparar_dados",
        "treinar_modelos",
        "avaliar_modelos",
        "segmentar_clientes",
        "predizer_clientes"
    ]

    for etapa in etapas:
        print(f"\n{'=' * 50}", flush=True)
        print(f"ETAPA: {etapa.upper()}", flush=True)
        print("=" * 50, flush=True)

        subprocess.run(
            [PYTHON, "-m", f"src.{etapa}"],
            cwd=ROOT,
            check=True
        )

    print("\nPIPELINE CONCLUÍDO!", flush=True)
    print("INICIANDO FASTAPI...", flush=True)

    subprocess.run(
        [PYTHON, "-m", "uvicorn", "src.api:app", "--reload"],
        cwd=ROOT,
        check=True
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
