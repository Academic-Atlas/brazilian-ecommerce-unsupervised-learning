
import subprocess
import sys

def main():
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
        print(f"\n{'=' * 50}")
        print(f"ETAPA: {etapa.upper()}")
        print('=' * 50)

        subprocess.run(
            [sys.executable, "-m", f"src.{etapa}"],
            check=True
        )

    print("\nPIPELINE CONCLUÍDO!")
    print("INICIANDO API FASTAPI...")

    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "src.api:app",
        "--reload"
    ])

if __name__ == "__main__":
    main()
