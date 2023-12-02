from api.app import app
import pytest


def test_get_root(): 
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert response.get_data() == b"<p>Hello, World!</p>"
    
def test_post_root():
    suffix ="post suffix"
    response = app.test_client().post("/", json={"suffix":suffix})
    assert response.status_code == 200
    assert response.get_json()['op'] =="Hello, World POST "+suffix

def test_sum_numbers():
    suffix ="post suffix"
    response = app.test_client().get("/inference", query_string={'a':2, 'b':4})
    assert response.status_code == 200
    assert response.get_data(as_text=True) == '6'