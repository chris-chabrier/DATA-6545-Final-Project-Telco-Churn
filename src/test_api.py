import requests

BASE_URL = "http://127.0.0.1:5000"

def print_result(test_name, passed, message=""):
    status = "PASS" if passed else "FAIL"
    print(f"{test_name}: {status}")
    if message:
        print(f"  → {message}")
    print("-" * 50)

# Test 1: Health Check
def test_health():
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()

        passed = (
            response.status_code == 200 and
            "status" in data and
            data["status"] == "ok"
        )

        print_result("Test 1 (/health)", passed, data)

    except Exception as e:
        print_result("Test 1 (/health)", False, str(e))

# Test 2: Single Prediction
def test_single_prediction():
    payload = {
        "tenure": 5,
        "MonthlyCharges": 85.5,
        "TotalCharges": 300,
        "Contract": "Month-to-month",
        "PhoneService": "Yes",
        "StreamingTV": "Yes"
    }

    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        data = response.json()

        passed = (
            response.status_code == 200 and
            "prediction" in data and
            "churn_probability" in data and
            "risk_label" in data
        )

        print_result("Test 2 (/predict)", passed, data)

    except Exception as e:
        print_result("Test 2 (/predict)", False, str(e))


# Test 3: Batch Prediction (5 records)
def test_batch_prediction():
    payload = [
        {
            "tenure": 5,
            "MonthlyCharges": 85.5,
            "TotalCharges": 300
        },
        {
            "tenure": 24,
            "MonthlyCharges": 60,
            "TotalCharges": 1500
        },
        {
            "tenure": 1,
            "MonthlyCharges": 95,
            "TotalCharges": 95
        },
        {
            "tenure": 36,
            "MonthlyCharges": 45,
            "TotalCharges": 1600
        },
        {
            "tenure": 12,
            "MonthlyCharges": 70,
            "TotalCharges": 800
        }
    ]

    try:
        response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
        data = response.json()

        passed = (
            response.status_code == 200 and
            "results" in data and
            len(data["results"]) == 5
        )

        print_result("Test 3 (/predict/batch)", passed, data)

    except Exception as e:
        print_result("Test 3 (/predict/batch)", False, str(e))


# Test 4: Missing Required Field
def test_missing_field():
    payload = {
        # Missing tenure
        "MonthlyCharges": 85.5,
        "TotalCharges": 300
    }

    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)

        passed = response.status_code == 400

        print_result("Test 4 (missing field)", passed, response.text)

    except Exception as e:
        print_result("Test 4 (missing field)", False, str(e))


# Test 5: Invalid Data Type
def test_invalid_type():
    payload = {
        "tenure": "invalid_string",  # should be numeric
        "MonthlyCharges": 85.5,
        "TotalCharges": 300
    }

    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)

        passed = response.status_code == 400

        print_result("Test 5 (invalid type)", passed, response.text)

    except Exception as e:
        print_result("Test 5 (invalid type)", False, str(e))

# Run All Tests
if __name__ == "__main__":
    print("\nRunning API Tests...\n")

    test_health()
    test_single_prediction()
    test_batch_prediction()
    test_missing_field()
    test_invalid_type()

    print("\nAll tests completed.\n")