import time
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

now = None

def tick():
    """
    Begins the stopwatch.
    
    """
    global now
    now = time.clock()


def tock(*s):
    """
    Prints a 'lap' of the stopwatch since last call to 'tick' or 'tock'.
    """
    global now
    assert now is not None, "Gotta tick before ya can tock!"
    if s:
        print(s[0], ':', round(time.clock() - now, 2), "seconds")
    tick()


def classify(params, x1, y1, x2, y2):
    """
    Performs GridSearch and training on a collection of classifiers and parameters. Calculates performance metrics for each.
    
    Parameters
    ----------
    params : A dictionary of classifiers and GridSearchCV parameters: {'classifier_name': <Classifier_instance>, {GridSearchCV params dictionary}}
    x1 : training dataset features
    y1 : training dataset labels
    x2 : testing dataset features
    y2 : testing dataset labels
    scorer : the metrics to optimize with GridSearch — typically a string like 'accuracy', 'precision', 'f1', etc.
    
    Returns
    -------
    A Python dictionary storing data on the trained and hyperparameter-tuned model.
    
    """
    now = time.clock()
    gscv = GridSearchCV(*params, scoring='r2').fit(x1, y1)
    t = time.clock() - now
    pred2 = gscv.predict(x2)
    return {
        'r2': r2_score(y2, pred2),
        'best_classifier': gscv.best_estimator_,
        'best_params': gscv.best_params_,
        'training time': t,
    }
