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

        # Initialize variables to collect result text, citations, and trace data
        output_text = ""
        citations = []
        trace = {"preProcessingTrace": [], "orchestrationTrace": [], "postProcessingTrace": []}

        # Process each chunk in the EventStream
        for event in response['completion']:
            chunk = event.get("chunk")
            if chunk and "bytes" in chunk:
                decoded_text = chunk["bytes"].decode().strip()
                
                # Extract only the 'result' field content
                match = re.search(r'"result":\s*"(.*?)"\s*}', decoded_text)
                if match:
                    output_text += match.group(1)
                else:
                    print("Warning: 'result' field not found in chunk:", decoded_text)

            # Collect citations from 'attribution' in each chunk
            if chunk and "attribution" in chunk:
                for citation in chunk["attribution"].get("citations", []):
                    if (
                        isinstance(citation, dict) and
                        "location" in citation and 
                        "s3Location" in citation["location"] and 
                        "uri" in citation["location"]["s3Location"]
                    ):
                        uri = citation["location"]["s3Location"]["uri"]
                        if uri not in citations:
                            citations.append(uri)

            # Collect trace information if available
            if "trace" in event:
                for trace_type in ["preProcessingTrace", "orchestrationTrace", "postProcessingTrace"]:
                    if trace_type in event["trace"]["trace"]:
                        trace[trace_type].append(event["trace"]["trace"][trace_type])

        # Remove placeholder markers like %[1]% if present
        output_text = re.sub(r'%\[\d+\]%', '', output_text).strip()

        # Append formatted citations if they exist
        if citations:
            citation_texts = "\n\nCitations:\n" + "\n".join([f"[{i+1}] {uri}" for i, uri in enumerate(citations)])
            output_text += citation_texts

        # Log final output_text for verification
        print("Final output_text:", output_text)

    except ClientError as e:
        output_text = "An error occurred while trying to invoke the agent."
        print(f"ClientError: {e}")

    return {
        "output_text": output_text,
        "citations": citations,
        "trace": trace
    }
