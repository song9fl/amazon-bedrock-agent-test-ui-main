import boto3
from botocore.exceptions import ClientError

def invoke_agent(agent_id, agent_alias_id, session_id, prompt):
    try:
        # Initialize the client for Bedrock Agent Runtime
        client = boto3.session.Session().client(service_name="bedrock-agent-runtime")
        
        # Invoke the agent
        response = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            enableTrace=True,
            sessionId=session_id,
            inputText=prompt,
        )

        output_text = ""
        citations = []
        trace = {}

        has_guardrail_trace = False
        for event in response.get("completion", []):
            # Combine chunks to get the output text
            if "chunk" in event:
                chunk = event["chunk"]
                output_text += chunk["bytes"].decode()
                
                # Collect citations from the chunk's attribution if available
                if "attribution" in chunk and "citations" in chunk["attribution"]:
                    citations += chunk["attribution"]["citations"]

            # Extract trace information from all events
            if "trace" in event:
                for trace_type in ["guardrailTrace", "preProcessingTrace", "orchestrationTrace", "postProcessingTrace"]:
                    if trace_type in event["trace"]["trace"]:
                        mapped_trace_type = trace_type
                        if trace_type == "guardrailTrace":
                            if not has_guardrail_trace:
                                has_guardrail_trace = True
                                mapped_trace_type = "preGuardrailTrace"
                            else:
                                mapped_trace_type = "postGuardrailTrace"
                        if mapped_trace_type not in trace:
                            trace[mapped_trace_type] = []
                        trace[mapped_trace_type].append(event["trace"]["trace"][trace_type])

        # Replace placeholder citations with hyperlinked markers and construct citation text
        citation_text = "\n"  # Collect citation links to append to the output
        for i, citation in enumerate(citations, start=1):
            citation_marker = f"[{i}]"
            hyperlink_marker = f"[{i}](#{i})"  # Markdown-style hyperlink within the text
            output_text = output_text.replace(f"%[{i}]%", hyperlink_marker)
            
            # Construct citation text with hyperlink to the citation's S3 location or relevant URI
            if "location" in citation and "s3Location" in citation["location"]:
                citation_url = citation["location"]["s3Location"]["uri"]
                citation_text += f"[{i}]: {citation_url}\n"
            else:
                citation_text += f"[{i}]: [Citation not available]\n"

        # Append all citations at the end of output_text as hyperlinks
        output_text += "\n" + citation_text

    except ClientError as e:
        print(f"An error occurred: {e}")
        raise

    return {
        "output_text": output_text,
        "citations": citations,
        "trace": trace
    }
