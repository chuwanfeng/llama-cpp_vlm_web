---
name: image_gen_best_practices
description: Best practices for using the image_generate tool to ensure successful and high-quality AI image creation.
version: 1.0.0
priority: 5
tools:
  - image_generate
---

# Image Generation Workflow

When using the `image_generate` tool, always ensure the following:
1. **Required Parameters**: The `prompt` argument is mandatory. Never call the tool without a prompt.
2. **Prompt Engineering**: Create descriptive, high-quality prompts. Instead of simple phrases, include details about:
   - **Subject**: (e.g., "A cute little cat")
   - **Setting/Environment**: (e.g., "sitting by the window")
   - **Lighting**: (e.g., "soft natural light", "golden hour")
   - **Atmosphere/Mood**: (e.g., "cozy", "serene")
   - **Technical Quality**: (e.g., "highly detailed", "8k resolution", "photorealistic")
3. **Error Handling**: If a 400 error occurs, analyze the prompt for potential issues and try refining the description or simplifying the request.