// export default ChatWithBotScreen;

import React, { useState, useEffect } from 'react';
import { View, Text, FlatList, StyleSheet, ActivityIndicator,Modal, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import ChatHeader from '@/components/RaidenChatHeader';

const ChatWithBotScreen = () => {
  interface Message {
    id: string;
    customer_request: string;
    agent_response: string;
  }

  const [botMessages, setBotMessages] = useState<Message[]>([]); 
  const [loading, setLoading] = useState(true);
  const [selectedMessage, setSelectedMessage] = useState<Message | null>(null);
  const [showPopup, setShowPopup] = useState(false);
  const [showSuccessPopup, setShowSuccessPopup] = useState(false);
  const router = useRouter();

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

  useEffect(() => {
    const fetchBotMessages = async () => {
      try {
        const response = await fetch('http://10.0.2.2:5000/dify-history');
        const data = await response.json();

        const formattedMessages = data.map((item: any) => ({
          id: item.message_id,
          customer_request: item.customer_request,
          agent_response: item.agent_response,
        }));

        setBotMessages(formattedMessages);
      } catch (error) {
        console.error('Error fetching bot messages:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchBotMessages();
  }, []);

  const handlePressMessage = (message: Message) => {
    setSelectedMessage(message); 
    setShowPopup(true);
  };
  
  const handleSendToCustomer = async () => {
    if (!selectedMessage) return;
  
    const accessToken = await getAccessToken();
    if (!accessToken) {
      console.error('Access token not found');
      return;
    }
  
    const payload = {
      recipient: {
        user_id: "1696434873920451916",
      },
      message: {
        text: selectedMessage.agent_response,
      },
    };
  
    try {
      const response = await fetch('https://openapi.zalo.me/v3.0/oa/message/cs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'access_token': accessToken,
        },
        body: JSON.stringify(payload),
      });
  
      if (response.ok) {
        console.log('Gửi tin nhắn thành công');
      } else {
        console.error('Gửi tin nhắn thất bại');
      }
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setShowPopup(false); 
      setShowSuccessPopup(true);
    }
  };
  
  const handleCloseSuccessPopup = () => {
    setShowSuccessPopup(false); 
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
      <ChatHeader title="Raiden" />
      <FlatList
        data={botMessages}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <TouchableOpacity onPress={() => handlePressMessage(item)}>
            <View>
              <View style={styles.customerMessageContainer}>
                <Text style={styles.customerMessageText}>{item.customer_request}</Text>
              </View>
              <View style={styles.agentMessageContainer}>
                <Text style={styles.agentMessageText}>{item.agent_response}</Text>
              </View>
            </View>
          </TouchableOpacity>
        )}
      />
      {showPopup && selectedMessage && (
        <Modal
          transparent={true}
          animationType="fade"
          visible={showPopup}
          onRequestClose={() => setShowPopup(false)}
        >
          <TouchableOpacity style={styles.overlay} onPress={() => setShowPopup(false)}>
            <View style={styles.popupContainer}>
              <Text style={styles.popupOption} onPress={handleSendToCustomer}>Gửi cho khách hàng</Text>
              <Text style={styles.popupOption} onPress={() => setShowPopup(false)}>Thoát</Text>
            </View>
          </TouchableOpacity>
        </Modal>
      )}

      {showSuccessPopup && (
        <Modal
          transparent={true}
          animationType="fade"
          visible={showSuccessPopup}
          onRequestClose={handleCloseSuccessPopup}
        >
          <TouchableOpacity style={styles.overlay} onPress={handleCloseSuccessPopup}>
            <View style={styles.successPopupContainer}>
              <Text style={styles.successText}>Gửi tin nhắn cho khách hàng thành công</Text>
            </View>
          </TouchableOpacity>
        </Modal>
      )}

    </View>
  );
};

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
  },
  popupContainer: {
    width: 200,
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
  customerMessageContainer: {
    padding: 10,
    backgroundColor: '#e1bee7', 
    borderRadius: 5,
    marginVertical: 5,
  },
  customerMessageText: { 
    fontSize: 16,
  },
  agentMessageContainer: {
    padding: 10,
    backgroundColor: '#f1f1f1', 
    borderRadius: 5,
    marginVertical: 5,
  },
  agentMessageText: {
    color: 'black', 
    fontSize: 16,
  },
  successPopupContainer: {
    width: 250,
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 20,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 5,
  },
  successText: {
    fontSize: 16,
    color: '#6200ee',
    marginTop: 10,
    textAlign: 'center',
  }
});

export default ChatWithBotScreen;
