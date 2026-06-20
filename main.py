from pipeline import run_pipeline

if __name__ == "__main__":
    outputs = run_pipeline("data/titanic.csv", target_column="Survived")
    for stage, text in outputs.items():
        print(f"\n{'='*60}\n{stage.upper()}\n{'='*60}\n{text}")
