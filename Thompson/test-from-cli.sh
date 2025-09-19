# pick an arm
curl http://localhost:8000/choose
# {"arm_id":"headline_B"}

# send reward
curl -X POST http://localhost:8000/reward \
     -H "Content-Type: application/json" \
     -d '{"arm_id":"headline_B","reward":1}'