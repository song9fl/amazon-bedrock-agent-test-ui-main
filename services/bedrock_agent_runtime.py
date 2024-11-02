import boto3
from botocore.exceptions import ClientError
import re

def invoke_agent(agent_id, agent_alias_id, session_id, prompt):
    try:
        # Initialize the client for the Bedrock agent runtime
        client = boto3.session.Session().client(service_name="bedrock-agent-runtime")
        
        # Invoke the agent with provided parameters
        response = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            enableTrace=True,
            sessionId=session_id,
            inputText=prompt,
        )

        # Extract 'result' as the main message
        output_text = response.get("result", "").strip()
        citations = []
        trace = response.get("trace", {})  # Ensure trace is always initialized

        # Clean up placeholder markers like %[1]%, %[2]% if present in 'result'
        output_text = re.sub(r'%\[\d+\]%', '', output_text)

        # If citations exist in 'completion' events, extract them
        for event in response.get("completion", []):
            chunk = event.get("chunk")
            if chunk and "attribution" in chunk:
                for citation in chunk["attribution"].get("citations", []):
                    # Ensure valid citation structure before accessing
                    if (
                        isinstance(citation, dict) and
                        "location" in citation and 
                        "s3Location" in citation["location"] and 
                        "uri" in citation["location"]["s3Location"]
                    ):
                        uri = citation["location"]["s3Location"]["uri"]
                        if uri not in citations:  # Avoid duplicates
                            citations.append(uri)

        # Format and append citations if they exist
        if citations:
            citation_texts = "\n\nCitations:\n" + "\n".join([f"[{i+1}] {uri}" for i, uri in enumerate(citations)])
            output_text += citation_texts.strip()

        # Log final output_text for verification
        print("Final output_text:", output_text)

    except ClientError as e:
        # Provide a user-friendly message in case of an error
        output_text = "An error occurred while trying to invoke the agent."
        print(f"ClientError: {e}")

    # Return the structured response with ensured 'trace' and 'citations' keys
    return {
        "output_text": output_text,
        "citations": citations,
        "trace": trace
    }
