import boto3
from botocore.exceptions import ClientError

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

        # Extract the main 'result' field for output, which is the bot's main response
        output_text = response.get("result", "")
        citations = []
        trace = {}

        # Only process 'completion' events if 'output_text' contains meaningful text
        if output_text:
            # Clean up any inline citation markers in the main response
            output_text = output_text.replace("%[1]%", "").replace("%[2]%", "").replace("%[3]%", "").replace("%[4]%", "").replace("%[5]%", "").strip()
        
            # Process each event in completion to gather citations
            for event in response.get("completion", []):
                chunk = event.get("chunk")
                if chunk and "attribution" in chunk:
                    # Collect citations separately
                    citations.extend(chunk["attribution"]["citations"])

                # Collect trace information if needed for debugging
                event_trace = event.get("trace")
                if event_trace:
                    for trace_type in ["preProcessingTrace", "orchestrationTrace", "postProcessingTrace"]:
                        trace_data = event_trace["trace"].get(trace_type)
                        if trace_data:
                            if trace_type not in trace:
                                trace[trace_type] = []
                            trace[trace_type].append(trace_data)

        # Format and append citations at the end of output_text if they exist
        if citations:
            citation_texts = "\n\nCitations:\n"
            for i, citation in enumerate(citations, start=1):
                uri = citation.get("location", {}).get("s3Location", {}).get("uri", "Citation unavailable")
                citation_texts += f"[{i}] {uri}\n"
            output_text += citation_texts.strip()

    except ClientError as e:
        # Handle client errors gracefully
        output_text = "An error occurred while trying to invoke the agent."
        print(f"ClientError: {e}")

    # Return the structured response
    return {
        "output_text": output_text,  # Main response text with appended citations if any
        "citations": citations,
        "trace": trace
    }
