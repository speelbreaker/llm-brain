#!/usr/bin/env python3
"""
Quick Telegram Bot Test Script

Usage:
    # Test bot token is valid
    python scripts/test_telegram.py --token YOUR_BOT_TOKEN
    
    # Get updates to find chat/topic IDs  
    python scripts/test_telegram.py --token YOUR_BOT_TOKEN --get-updates
    
    # Send a test message
    python scripts/test_telegram.py --token YOUR_BOT_TOKEN --chat-id CHAT_ID --message "Hello!"
    
    # Send to specific topic in supergroup
    python scripts/test_telegram.py --token YOUR_BOT_TOKEN --chat-id CHAT_ID --topic-id TOPIC_ID --message "Hello!"
"""

import argparse
import json
import sys

import requests


def test_bot(token: str) -> bool:
    """Test if bot token is valid."""
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("ok"):
            bot = data["result"]
            print(f"✅ Bot is valid!")
            print(f"   Username: @{bot['username']}")
            print(f"   Name: {bot['first_name']}")
            print(f"   Bot ID: {bot['id']}")
            return True
        else:
            print(f"❌ Invalid bot token: {data.get('description', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Error testing bot: {e}")
        return False


def get_updates(token: str) -> None:
    """Get recent updates to find chat/topic IDs."""
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        
        if not data.get("ok"):
            print(f"❌ Error: {data.get('description', 'Unknown error')}")
            return
        
        updates = data.get("result", [])
        if not updates:
            print("📭 No updates yet.")
            print("   Send a message to the bot or add it to a group, then run this again.")
            return
        
        print(f"📬 Found {len(updates)} update(s):\n")
        
        seen_chats = {}
        for update in updates:
            msg = update.get("message") or update.get("channel_post") or {}
            chat = msg.get("chat", {})
            chat_id = chat.get("id")
            chat_type = chat.get("type", "unknown")
            chat_title = chat.get("title") or chat.get("username") or chat.get("first_name") or "?"
            
            if chat_id and chat_id not in seen_chats:
                seen_chats[chat_id] = {
                    "type": chat_type,
                    "title": chat_title,
                }
                
                print(f"Chat: {chat_title}")
                print(f"  ID: {chat_id}")
                print(f"  Type: {chat_type}")
                
                # Check for topic
                thread_id = msg.get("message_thread_id")
                if thread_id:
                    print(f"  Topic ID: {thread_id}")
                
                print()
        
        print("=" * 50)
        print("📋 Copy these values to your .env.trading:")
        for chat_id, info in seen_chats.items():
            if info["type"] in ("supergroup", "group"):
                print(f"   TELEGRAM_SUPERGROUP_ID={chat_id}")
            elif info["type"] == "private":
                print(f"   TELEGRAM_CHAT_ID={chat_id}")
        
    except Exception as e:
        print(f"❌ Error getting updates: {e}")


def send_message(token: str, chat_id: str, message: str, topic_id: int = None) -> bool:
    """Send a test message."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
    }
    
    if topic_id:
        payload["message_thread_id"] = topic_id
    
    try:
        resp = requests.post(url, json=payload, timeout=10)
        data = resp.json()
        
        if data.get("ok"):
            print(f"✅ Message sent successfully!")
            msg = data["result"]
            print(f"   Message ID: {msg['message_id']}")
            if topic_id:
                print(f"   Topic ID: {topic_id}")
            return True
        else:
            print(f"❌ Failed to send: {data.get('description', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error sending message: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Telegram Bot Connection")
    parser.add_argument("--token", required=True, help="Bot token from @BotFather")
    parser.add_argument("--get-updates", action="store_true", help="Get updates to find chat IDs")
    parser.add_argument("--chat-id", help="Chat ID to send message to")
    parser.add_argument("--topic-id", type=int, help="Topic ID for supergroup threads")
    parser.add_argument("--message", default="🤖 Trading Loop test message!", help="Message to send")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🔧 Telegram Bot Test")
    print("=" * 50)
    print()
    
    # Always test the bot first
    if not test_bot(args.token):
        sys.exit(1)
    
    print()
    
    # Get updates if requested
    if args.get_updates:
        print("-" * 50)
        get_updates(args.token)
    
    # Send message if chat_id provided
    if args.chat_id:
        print("-" * 50)
        send_message(args.token, args.chat_id, args.message, args.topic_id)


if __name__ == "__main__":
    main()


