## FrontEnd-Demo

This is a plain front end built for demo of recommendation engine only. 

### Instruction

* Launch backend - Go to `Thompson` directory, and launch the server:

```python
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

* Launch frontend - Use current directory, and launch the frontend server:

```node
npx http-server -p 8080
```

or using `Python`

```python
python3 -m http.server 8080
```

Either way, index.html will be launched as the frontend.

* Login - Pick one of the users


