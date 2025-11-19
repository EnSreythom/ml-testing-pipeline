from ml_project.pipeline import load_data, train_model, evaluate_model

def test_load_data():
    df = load_data()
    assert not df.empty
    assert 'feature' in df.columns and 'target' in df.columns

def test_train_model():
    df = load_data()
    model = train_model(df[['feature']], df['target'])
    assert hasattr(model, 'predict')

def test_evaluate_accuracy():
    df = load_data()
    model = train_model(df[['feature']], df['target'])
    preds = model.predict(df[['feature']])
    result = evaluate_model(df['target'], preds)
    assert 0.0 <= result['accuracy'] <= 1.0