# LLM Learning NOTES part 5
```cmd
venv\Scripts\activate
jupyter lab
```

## Week 8 **GAME ON**

### What you can now do
* Generate text and code with Frontier Models and Open Source models using APIs and HuggingFace, including tools, assistants and RAG
* Follow a 5 step strategy to solve problems, including dataset curation, making a baseline model, and fine-tuning a Frontier model
* Confidently carry out the end-to-end process for selecting and training specialized open source models that can outperform the Frontier

---

### An intensive week lies ahead! After today, you’ll be able to:
* Use Modal, the serverless platform for AI, to run code remotely
* Deploy an LLM behind an API in the cloud
* Create an Agent that will be incorporated into an autonomous Agentic AI solution

```code
modal setup
    The web browser should have opened for you to authenticate and get an API token.
    If it didn't, please copy this URL into your web browser manually:

    https://modal.com/token-flow/tf-Uf2WoGWXSSfgQooe1bbBPl

    ⠏ Waiting for token flow to complete...
    Web authentication finished successfully!
    Token is connected to the samfire5200 workspace.
    Verifying token against https://api.modal.com
    Token verified successfully!
    Token written to C:\Users\samfi/.modal.toml in profile samfire5200.      

```
## We need to set your HuggingFace Token as a secret in Modal

1. Go to modal.com, sign in and go to your dashboard
2. Click on Secrets in the nav bar
3. Create new secret, click on Hugging Face, this new secret needs to be called **hf-secret** because that's how we refer to it in the code
4. Fill in your HF_TOKEN where it prompts you
