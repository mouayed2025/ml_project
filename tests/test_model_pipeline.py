from main import prepare_data, train_model


def test_prepare_data_shapes():
    x_train, x_test, y_train, y_test = prepare_data()
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert x_train.shape[1] == 9


def test_train_model_predicts_0_or_1():
    x_train, x_test, y_train, y_test = prepare_data()
    model = train_model(x_train, y_train)
    predictions = model.predict(x_test)
    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})
