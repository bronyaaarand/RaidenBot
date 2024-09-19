// // // app/chat/person.tsx

import React, { useState, useEffect } from 'react';
import { View, Text, FlatList, StyleSheet, ActivityIndicator, Modal, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import ChatHeader from '@/components/RaidenChatHeader';

const ChatWithPersonScreen = () => {
  interface Message {
    id: string;
    text: string;
    from_id: string;
  }

  const [personMessages, setPersonMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedMessage, setSelectedMessage] = useState<Message | null>(null); // Track the selected message
  const [showPopup, setShowPopup] = useState(false); // Show/hide popup
  const router = useRouter();

  const targetFromId = "2369284118391448762"; // id cua Gcalls OA

  const getAccessToken = async () => {
    try {
      const response = await fetch('http://10.0.2.2:5000/access-token');
      const data = await response.json();
      
      if (data.access_token) {
        return data.access_token;
      } else {
        console.error('Access token not found in response');
        return null;
      }
    } catch (error) {
      console.error('Error fetching access token from backend:', error);
      return null;
    }
  };

  interface MessageData {
    message_id: string;
    message: string;
    from_id: string;
  }

  useEffect(() => {
    const fetchMessages = async () => {
      try {
        const accessToken = await getAccessToken();
        if (!accessToken) {
          console.error('Access token not found');
          setLoading(false);
          return;
        }

        const response = await fetch(
          'https://openapi.zalo.me/v2.0/oa/conversation?data={"user_id":1696434873920451916,"offset":0,"count":10}', 
          {
            method: 'GET',
            headers: {
              'access_token': accessToken,
            },
          }
        );

        const result = await response.json();
        const messages = result.data.map((msg: MessageData) => ({
          id: msg.message_id,
          text: msg.message,
          from_id: msg.from_id,
        }));

        setPersonMessages(messages);

      } catch (error) {
        console.error('Error fetching messages:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchMessages();
  }, []);

  const handlePressMessage = (message: Message) => {
    setSelectedMessage(message); 
    setShowPopup(true); // Show pop-up
  };

  const handleSendToAI = () => {
    if (selectedMessage) {
      console.log(`Sending message to AI: ${selectedMessage.text}`);
      // Perform API call to send message to AI
    }
    setShowPopup(false); // Close the pop-up
  };

  const handleClosePopup = () => {
    setShowPopup(false); // Close the pop-up
  };

  const getMessageStyle = (from_id: string) => {
    if (from_id === targetFromId) {
      return [styles.messageContainer, styles.messageGray]; // Gray message style for specific from_id
    }
    return [styles.messageContainer, styles.messagePurple]; // Purple message for others
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#6200ee" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <ChatHeader title="Chat with MinhVu" />
      <FlatList
        data={personMessages}
        keyExtractor={(item) => item.id.toString()}
        renderItem={({ item }) => (
          <TouchableOpacity onPress={() => handlePressMessage(item)}>
            <View style={getMessageStyle(item.from_id)}>
              <Text style={styles.messageText}>{item.text}</Text>
            </View>
          </TouchableOpacity>
        )}
      />

      {/* Pop-up Modal */}
      {showPopup && selectedMessage && (
        <Modal
          transparent={true}
          animationType="fade"
          visible={showPopup}
          onRequestClose={handleClosePopup}
        >
          <TouchableOpacity style={styles.overlay} onPress={handleClosePopup}>
            <View style={styles.popupContainer}>
              <Text style={styles.popupOption} onPress={handleSendToAI}>Gửi cho AI</Text>
              <Text style={styles.popupOption} onPress={handleClosePopup}>Thoát</Text>
            </View>
          </TouchableOpacity>
        </Modal>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'white',
    padding: 10,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  messageContainer: {
    padding: 10,
    borderRadius: 5,
    marginVertical: 5,
  },
  messageGray: {
    backgroundColor: '#f1f1f1',
    color: 'black',
  },
  messagePurple: {
    backgroundColor: '#e1bee7',
    color: 'white',
    fontWeight: 'bold',
  },
  messageText: {
    fontSize: 16,
  },
  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
  },
  popupContainer: {
    width: 150,
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 10,
    shadowColor: '#000',
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 5,
  },
  popupOption: {
    fontSize: 18,
    padding: 10,
    textAlign: 'center',
    borderBottomColor: '#ddd',
    borderBottomWidth: 1,
    color: '#6200ee',
  },
});

export default ChatWithPersonScreen;
