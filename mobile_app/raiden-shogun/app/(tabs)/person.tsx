// app/chat/person.tsx

import React from 'react';
import { View, Text, FlatList, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import ChatHeader from '@/components/RaidenChatHeader';

const personMessages = [
  { id: '1', text: 'Xin chào tôi muốn được hỗ trợ trực tiếp !' },
  { id: '2', text: 'Xin chào tôi muốn được hỗ trợ trực tiếp !' },
  { id: '3', text: 'Xin chào tôi muốn được hỗ trợ trực tiếp !' },
  { id: '4', text: 'Xin chào tôi muốn được hỗ trợ trực tiếp !' },
  { id: '5', text: 'Xin chào tôi muốn được hỗ trợ trực tiếp !' },
  { id: '6', text: 'Xin chào tôi muốn được hỗ trợ trực tiếp !' },
  { id: '7', text: 'Xin chào tôi muốn được hỗ trợ trực tiếp !' },
  { id: '8', text: 'Xin chào tôi muốn được hỗ trợ trực tiếp !' },
  { id: '9', text: 'Xin chào tôi muốn được hỗ trợ trực tiếp !' },
  { id: '10', text: 'Xin chào tôi muốn được hỗ trợ trực tiếp !' },
];

const ChatWithPersonScreen = () => {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <ChatHeader title="Khách hàng" />
      <FlatList
        data={personMessages}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <View style={styles.messageContainer}>
            <Text style={styles.messageText}>{item.text}</Text>
          </View>
        )}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'white',
    padding: 10,
  },
  messageContainer: {
    padding: 10,
    backgroundColor: '#f1f1f1',
    borderRadius: 5,
    marginVertical: 5,
  },
  messageText: {
    fontSize: 16,
  },
});

export default ChatWithPersonScreen;
