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

        # Extract the chatbot's main message from 'result'
        output_text = response.get("result", "").strip()
        citations = []

        # Remove inline placeholder markers like %[1]%, %[2]% from the main message
        output_text = re.sub(r'%\[\d+\]%', '', output_text).strip()

        # Process 'completion' events to gather citations
        for event in response.get("completion", []):
            chunk = event.get("chunk")
            if chunk and "attribution" in chunk:
                for citation in chunk["attribution"].get("citations", []):
                    # Ensure citation structure is valid before accessing
                    if (
                        isinstance(citation, dict) and
                        "location" in citation and 
                        "s3Location" in citation["location"] and 
                        "uri" in citation["location"]["s3Location"]
                    ):
                        uri = citation["location"]["s3Location"]["uri"]
                        if uri not in citations:  # Avoid duplicate URIs
                            citations.append(uri)

        # Append formatted citations at the end of output_text if citations exist
        if citations:
            citation_texts = "\n\nCitations:\n" + "\n".join([f"[{i+1}] {uri}" for i, uri in enumerate(citations)])
            output_text += citation_texts

    except ClientError as e:
        # Handle client errors gracefully
        output_text = "An error occurred while trying to invoke the agent."
        print(f"ClientError: {e}")

    # Return the structured response with ensured 'citations' key
    return {
        "output_text": output_text,
        "citations": citations,  # Always include citations, even if empty
    }
