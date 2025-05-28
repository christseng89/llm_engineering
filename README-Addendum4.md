# LLM Learning NOTES part 4
```cmd
venv\Scripts\activate
jupyter lab
```

## Week 7 - Fine-Tuned Open Source LoRa and QLoRA Models
### Week 7 Day 1
#### Learning Objectives
- Explain LoRA for fine-tuning Open Source models
- Describe Quantization and QLoRA
- Explain 3 key hyper-parameters: r, alpha and target modules

🧠 Summary Table
| Hyperparameter   | Meaning                          | Typical Value  | Notes                         |
| ---------------- | -------------------------------- | -------------- | ----------------------------- |
| `r`              | Rank of low-rank matrices        | 4–16           | Controls parameter efficiency |
| `alpha`          | Scaling factor for LoRA update   | 16–32          | Higher = more influence       |
| `target_modules` | Model layers to inject LoRA into | model-specific | Must match architecture       |
