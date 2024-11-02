import boto3
from botocore.exceptions import ClientError
import re
import json

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

        # Initialize variables to collect result text and citations
        output_text = ""
        citations = []
        trace = response.get("trace", {})  # Initialize trace

        # Process each chunk in the EventStream, focusing on 'result'
        for event in response['completion']:
            chunk = event.get("chunk")
            if chunk and "bytes" in chunk:
                # Decode and parse as JSON to isolate 'result' only
                decoded_text = chunk["bytes"].decode().strip()
                try:
                    # Parse each chunk as JSON and accumulate only the 'result' field
                    chunk_json = json.loads(decoded_text)
                    output_text += chunk_json.get("result", "")
                except json.JSONDecodeError:
                    print("Error: Unable to parse chunk as JSON:", decoded_text)

        # Clean up placeholder markers like %[1]% if present
        output_text = re.sub(r'%\[\d+\]%', '', output_text).strip()

        # Collect citations if available in each chunk's attribution
        for event in response.get("completion", []):
            chunk = event.get("chunk")
            if chunk and "attribution" in chunk:
                for citation in chunk["attribution"].get("citations", []):
                    if (
                        isinstance(citation, dict) and
                        "location" in citation and 
                        "s3Location" in citation["location"] and 
                        "uri" in citation["location"]["s3Location"]
                    ):
                        uri = citation["location"]["s3Location"]["uri"]
                        if uri not in citations:  # Avoid duplicates
                            citations.append(uri)

        # Append formatted citations at the end of output_text if any exist
        if citations:
            citation_texts = "\n\nCitations:\n" + "\n".join([f"[{i+1}] {uri}" for i, uri in enumerate(citations)])
            output_text += citation_texts

        # Log final output_text for verification
        print("Final output_text:", output_text)

    except ClientError as e:
        # Provide an error message if invocation fails
        output_text = "An error occurred while trying to invoke the agent."
        print(f"ClientError: {e}")

    # Return the structured response with both 'trace' and 'citations' ensured
    return {
        "output_text": output_text,
        "citations": citations,
        "trace": trace
    }
