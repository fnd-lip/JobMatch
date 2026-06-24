# JobMatch AI

Sistema de recomendação de vagas usando NLP e Machine Learning.

O projeto recebe um perfil profissional ou currículo em texto livre e retorna vagas com maior aderência ao candidato.

## Funcionalidades

* Recomendação Top-5 de vagas
* Score de aderência entre candidato e vaga
* Classificação Fit ou No Fit
* Identificação de skills compatíveis e faltantes
* Sugestão de desenvolvimento profissional
* Estimativa salarial
* Registro de métricas e artefatos com MLflow/DagsHub

## Técnicas utilizadas

* NLP
* TF-IDF
* Similaridade por cosseno
* Modelo supervisionado para classificação Fit/No Fit
* Modelo de regressão para estimativa salarial
* MLflow para rastreamento de experimentos

## Validação

```powershell
python -m ruff check src tests
python -m src.pipeline.registrar_modelo_mlflow
```

## Interface

```powershell
streamlit run src\ui\streamlit_app.py
```

