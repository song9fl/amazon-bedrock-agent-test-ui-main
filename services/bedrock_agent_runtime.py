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

        # Initialize output_text and citations to build the response
        output_text = ""
        citations = []
        trace = response.get("trace", {})  # Initialize trace to an empty dictionary by default

        # Process the EventStream in 'completion' to extract response data
        for event in response['completion']:
            # Decode each event chunk as it arrives
            chunk = event.get("chunk")
            if chunk and "bytes" in chunk:
                # Decode the response text and append it to output_text
                output_text += chunk["bytes"].decode().strip()
            
            # Collect any citations if they exist in the chunk
            if chunk and "attribution" in chunk:
                for citation in chunk["attribution"].get("citations", []):
                    if (
                        isinstance(citation, dict) and
                        "location" in citation and 
                        "s3Location" in citation["location"] and 
                        "uri" in citation["location"]["s3Location"]
                    ):
                        uri = citation["location"]["s3Location"]["uri"]
                        if uri not in citations:  # Avoid duplicate URIs
                            citations.append(uri)

        # Clean up any placeholder markers like %[1]% if present
        output_text = re.sub(r'%\[\d+\]%', '', output_text).strip()

        # Append formatted citations at the end of output_text if they exist
        if citations:
            citation_texts = "\n\nCitations:\n" + "\n".join([f"[{i+1}] {uri}" for i, uri in enumerate(citations)])
            output_text += citation_texts

        # Log final output_text for verification
        print("Final output_text:", output_text)

    except ClientError as e:
        # Provide a user-friendly message in case of an error
        output_text = "An error occurred while trying to invoke the agent."
        print(f"ClientError: {e}")

    # Return the structured response with ensured 'trace' and 'citations' keys
    return {
        "output_text": output_text,
        "citations": citations,  # Always include citations, even if empty
        "trace": trace  # Ensure trace key is always present
    }
