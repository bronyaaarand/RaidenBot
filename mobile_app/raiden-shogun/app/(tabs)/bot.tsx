// app/chat/bot.tsx

import React, { useState, useEffect } from 'react';
import { View, Text, FlatList, StyleSheet, ActivityIndicator  } from 'react-native';
import { useRouter } from 'expo-router';
import ChatHeader from '@/components/RaidenChatHeader';
import { MongoClient } from 'mongodb';

const botMessages = [
  { id: '1', text: 'Đây là tin nhắn từ Zalo !' },
  { id: '2', text: 'Đây là tin nhắn từ Zalo !' },
  { id: '3', text: 'Đây là tin nhắn từ Zalo !' },
  { id: '4', text: 'Đây là tin nhắn từ Zalo !' },
  { id: '5', text: 'Đây là tin nhắn từ Zalo !' },
  { id: '6', text: 'Đây là tin nhắn từ Zalo !' },
  { id: '7', text: 'Đây là tin nhắn từ Zalo !' },
  { id: '8', text: 'Đây là tin nhắn từ Zalo !' },
  { id: '9', text: 'Đây là tin nhắn từ Zalo !' },
  { id: '10', text: 'Đây là tin nhắn từ Zalo !' },
];

const ChatWithBotScreen = () => {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <ChatHeader title="Raiden" />
      <FlatList
        data={botMessages}
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

export default ChatWithBotScreen;
