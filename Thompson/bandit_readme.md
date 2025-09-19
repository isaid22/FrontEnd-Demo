## Thompson Sampling

Below is a drop-in Thompson Sampling bandit that plugs straight into your FastAPI backend on localhost:8000.
It keeps everything in memory (no Redis) so you can test in a single Python process; swap the MemoryStore for Redis later.

### Prerequisite
```python
pip install fastapi uvicorn numpy
```

### Next steps

Run 

```python
uvicorn main:app --reload --port 8000
```

Test from CLI (or HTML page)
```
# pick an arm
curl http://localhost:8000/choose
# {"arm_id":"headline_B"}

# send reward
curl -X POST http://localhost:8000/reward \
     -H "Content-Type: application/json" \
     -d '{"arm_id":"headline_B","reward":1}'
     
```

Wire into your HTML (vanilla JS)

```html
<button id="getBtn">Get recommendation</button>
<div id="rec"></div>

<script>
async function getRec(){
  const res = await fetch("http://localhost:8000/choose");
  const json = await res.json();
  document.getElementById("rec").innerText = "Show: " + json.arm_id;
  // store arm_id so you can report reward later
  localStorage.setItem("lastArm", json.arm_id);
}
document.getElementById("getBtn").onclick = getRec;

// example: user clicks “Buy” → report reward=1
function reportReward(reward){
  const arm = localStorage.getItem("lastArm");
  fetch("http://localhost:8000/reward", {
    method: "POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({arm_id: arm, reward})
  });
}
</script>
```


Next steps when you outgrow memory

1. Replace the dict inside ThompsonBandit with a tiny Redis helper (same interface).
2. Persist every reward to disk/S3 for offline analysis.
3. Add /reset endpoint to clear priors between experiments.

That’s it—your FastAPI now serves Thompson-sampling recommendations in ~60 lines of code.