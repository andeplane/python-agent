from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from typing import Optional

client = OpenAI()

def call_openai(
          message_history: list[ChatCompletionMessageParam], 
          prompt: str, 
          system_prompt: str,
          model: str,
          ) -> Optional[str]:
        """
        Calls the OpenAI API with the given prompt and returns the response.

        Args:
            prompt (str): The prompt to send to the OpenAI API.

        Returns:
            Optional[str]: The API's response or None if failed.
        """
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    *message_history,
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            if response and response.choices:
                if not response.choices[0].message.content:
                    raise ValueError("OpenAI API response is empty.")
                llm_response = response.choices[0].message.content.strip()
                # if self.debug:
                #     print(f"Prompt Sent:\n{prompt}\n")
                #     print(f"LLM Response:\n{llm_response}\n")
                return llm_response
        except Exception as e:
            print(f"OpenAI API call failed: {str(e)}")
        return None
    