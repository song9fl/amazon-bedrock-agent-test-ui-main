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

        # Initialize variables to hold output text and citation information
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

        # Prepare citation markers and links
        citation_text = "\n"  # Collect citation links to append to the output
        for i, citation in enumerate(citations, start=1):
            citation_marker = f"[{i}]"
            # Check if 'location' and 's3Location' keys are present in the citation
            if "location" in citation and "s3Location" in citation["location"]:
                citation_url = citation["location"]["s3Location"]["uri"]
                hyperlink_marker = f"[{i}]({citation_url})"  # Markdown-style hyperlink
                citation_text += f"{citation_marker}: {citation_url}\n"
            else:
                hyperlink_marker = f"[{i}](#)"  # Placeholder link if citation location is missing
                citation_text += f"{citation_marker}: [Citation not available]\n"
            
            # Replace placeholders in the output_text
            output_text = output_text.replace(f"%[{i}]%", hyperlink_marker)

        # Append all citations at the end of output_text
        output_text += "\n" + citation_text

    except ClientError as e:
        print(f"An error occurred: {e}")
        raise

    # Return only the formatted output text without extra metadata like instruction
    return {
        "output_text": output_text,
        "trace": trace  # Keep trace for debugging if needed; can be omitted if not needed
    }
