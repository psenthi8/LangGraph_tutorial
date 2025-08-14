from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama, OllamaEmbeddings
import json
import re

load_dotenv()

# This is the global variable to store document content
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"


@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """
    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"


def create_email_content(user_input: str) -> str:
    """Creates email content based on user input - fallback function"""
    
    # Extract recipient name
    recipient_match = re.search(r'to\s+(\w+)', user_input.lower())
    recipient = recipient_match.group(1).title() if recipient_match else "Friend"
    
    # Extract context
    context = ""
    if "evening" in user_input.lower() and "hanging out" in user_input.lower():
        context = "this evening for hanging out"
    elif "today" in user_input.lower():
        context = "today"
    elif "join" in user_input.lower():
        context = "join you as planned"
    else:
        context = "our planned meeting"
    
    # Create the email
    email_content = f"""Subject: Unable to Meet {context.split()[-1] if context else 'Today'}

Hi {recipient},

I hope you're doing well. I'm writing to let you know that I won't be able to {context}.

I apologize for any inconvenience this may cause. Let's catch up later when I'm available.

Thanks for understanding!

Best regards"""
    
    return email_content


tools = [update, save]

# Try different model configurations
try:
    # First try with different parameters
    model = ChatOllama(
        model="llama3.2", 
        temperature=0.1,
        num_predict=1024,  # Increase significantly
        top_k=10,
        top_p=0.95,
        repeat_penalty=1.15,
        stop=[]  # Empty stop list
    ).bind_tools(tools)
    print("✅ Using Llama 3.2 with enhanced parameters")
except Exception as e:
    print(f"❌ Ollama setup failed: {e}")
    # Fallback to OpenAI if available
    try:
        model = ChatOpenAI(model="gpt-4o").bind_tools(tools)
        print("✅ Fallback to OpenAI GPT-4")
    except Exception as e2:
        print(f"❌ OpenAI also failed: {e2}")
        raise Exception("No working model available")


def our_agent(state: AgentState) -> AgentState:
    global document_content
    
    # Much more explicit system prompt
    system_prompt = SystemMessage(content=f"""
You are Drafter, a helpful writing assistant. Help users create and modify documents.

CURRENT DOCUMENT CONTENT: {document_content if document_content else "No content yet."}

TOOL USAGE RULES:
1. For creating/updating content: Use 'update' tool with COMPLETE content as parameter
2. For saving: Use 'save' tool with filename

IMPORTANT: When creating emails, write the COMPLETE email including:
- Subject line
- Greeting
- Full message body  
- Closing
- Signature

Example email format:
Subject: [Topic]

Hi [Name],

[Complete message explaining the situation]

[Appropriate closing]

Best regards

DO NOT cut off or truncate content. Provide the FULL email text.
""")

    if not state["messages"]:
        user_input = "I'm ready to help you create or update a document. What would you like to do?"
        user_message = AIMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    
    # Show debug info
    print("\n" + "="*50)
    print("SYSTEM PROMPT:")
    print(system_prompt.content[:200] + "...")
    print("USER INPUT:", user_input if 'user_input' in locals() else "Initial message")
    print("="*50)

    try:
        response = model.invoke(all_messages)
        
        print(f"\n🔧 RAW RESPONSE CONTENT: '{response.content}'")
        print(f"🔧 HAS TOOL CALLS: {hasattr(response, 'tool_calls') and bool(response.tool_calls)}")
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"🔧 TOOL CALLS: {response.tool_calls}")
            
            # Check for the specific truncation issue
            for i, tool_call in enumerate(response.tool_calls):
                if (tool_call['name'] == 'update' and 
                    'content' in tool_call['args'] and 
                    tool_call['args']['content'].endswith('I won')):
                    
                    print("⚠️ DETECTED TRUNCATION ISSUE - APPLYING FALLBACK")
                    
                    # Create proper email content using fallback
                    if isinstance(user_message, HumanMessage):
                        full_email = create_email_content(user_message.content)
                        response.tool_calls[i]['args']['content'] = full_email
                        print(f"✅ FALLBACK CONTENT APPLIED:\n{full_email}")
                    
        elif ('email' in user_input.lower() or 'mail' in user_input.lower()) and isinstance(user_message, HumanMessage):
            # If no tool calls but user wants email, force create one
            print("⚠️ NO TOOL CALL DETECTED BUT EMAIL REQUESTED - CREATING FALLBACK")
            full_email = create_email_content(user_message.content)
            
            # Create a new response with proper tool call
            response = AIMessage(
                content=f"I'll create that email for you.",
                tool_calls=[{
                    'name': 'update',
                    'args': {'content': full_email},
                    'id': 'fallback_email_call',
                    'type': 'tool_call'
                }]
            )
            print(f"✅ FALLBACK RESPONSE CREATED")
        
        print(f"\n🤖 AI: {response.content}")
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"🔧 FINAL TOOLS: {[tc['name'] for tc in response.tool_calls]}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        
        # Emergency fallback
        if isinstance(user_message, HumanMessage) and ('email' in user_message.content.lower() or 'mail' in user_message.content.lower()):
            print("🚨 EMERGENCY FALLBACK - CREATING EMAIL DIRECTLY")
            full_email = create_email_content(user_message.content)
            
            response = AIMessage(
                content=f"I've created the email for you (using emergency fallback):\n\n{full_email}",
                tool_calls=[{
                    'name': 'update',
                    'args': {'content': full_email},
                    'id': 'emergency_fallback',
                    'type': 'tool_call'
                }]
            )
        else:
            response = AIMessage(content=f"I encountered an error: {e}. Please try again.")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""
    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # Check for save tool message
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end"
        
    return "continue"

def print_messages(messages):
    """Function made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


# Graph setup
graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()

def run_document_agent():
    print("\n ===== ROBUST DRAFTER =====")
    print("This version includes fallback logic for truncation issues")
    
    state = {"messages": []}
    
    try:
        for step in app.stream(state, stream_mode="values"):
            if "messages" in step:
                print_messages(step["messages"])
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Application error: {e}")
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()