# Deployment V2 Isaac binding

Unfrozen V2-B engineering package. It binds the frozen seven-dimensional observation and three-dimensional per-side planner action to the model_21800 110D target slices. It contains no optimizer, backward call, SAC update, or PPO update.

The current package deliberately separates pure contract checks from the existing Stage5 Isaac executor. An integrated Isaac B0/B1 runner must demonstrate lifecycle execution and physical contact before training readiness can be granted.
