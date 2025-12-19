import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import load_and_preprocess

def main():
    X, y = load_and_preprocess("../data/nsl_kdd.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Intrusion Detection Confusion Matrix")
    plt.tight_layout()
    plt.savefig("../results/confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()