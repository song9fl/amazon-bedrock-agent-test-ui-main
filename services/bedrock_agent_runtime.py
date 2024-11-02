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

        # Initialize output_text to collect message chunks and an empty citations list
        output_text = ""
        citations = []
        trace = response.get("trace", {})  # Ensure trace is initialized

        # Process each chunk from the 'completion' EventStream
        for event in response['completion']:
            chunk = event.get("chunk")
            if chunk and "bytes" in chunk:
                # Decode and accumulate the text chunks for the main response
                output_text += chunk["bytes"].decode().strip()

        # Attempt to parse output_text as JSON and extract only the 'result' field
        try:
            response_json = json.loads(output_text)
            output_text = response_json.get("result", "").strip()  # Extract main response text
        except json.JSONDecodeError:
            print("Warning: output_text is not in JSON format.")
        
        # Clean up placeholder markers like %[1]% if present
        output_text = re.sub(r'%\[\d+\]%', '', output_text)

        # Collect citations from the completion stream if available
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

    # Return output with both 'trace' and 'citations' keys ensured
    return {
        "output_text": output_text,
        "citations": citations,
        "trace": trace
    }
