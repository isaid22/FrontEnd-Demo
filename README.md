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


* Check for arm reward update - Go to `Thompson/bandit_backup` directory and run diff between any two `json` files:

```
diff bandit_state_20250918_222841.json bandit_state_20250918_222902.json
```

For example: 

```
<     "timestamp": "2025-09-18T22:28:32.881722",
---
>     "timestamp": "2025-09-18T22:28:41.208930",
53c53
<             "alpha": 5.0,
---
>             "alpha": 6.0,
55,56c55,56
<             "total_reward": 4.0,
<             "num_pulls": 4
---
>             "total_reward": 5.0,
>             "num_pulls": 5
```

Notice the updated values shown by `diff`.