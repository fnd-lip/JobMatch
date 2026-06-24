from __future__ import annotations

from src.observability.mlflow_dagshub import registrar_execucao_mlflow
from src.pipeline.artefatos import carregar_artefatos
from src.pipeline.validacao_artefatos import (
    salvar_relatorio_validacao,
    validar_artefatos,
)


def exibir_resultado(inconsistencias: list[str]) -> None:
    if inconsistencias:
        print("Validação concluída com inconsistências.")

        for inconsistencia in inconsistencias:
            print("-", inconsistencia)

        return

    print("Validação concluída sem inconsistências.")


def main() -> None:
    artefatos = carregar_artefatos()
    inconsistencias = validar_artefatos(artefatos)
    caminho_relatorio = salvar_relatorio_validacao(artefatos, inconsistencias)

    registrar_execucao_mlflow(
        artefatos=artefatos,
        inconsistencias=inconsistencias,
        caminho_relatorio=caminho_relatorio,
    )

    exibir_resultado(inconsistencias)
    print("Registro no MLflow/DagsHub concluído.")


if __name__ == "__main__":
    main()
