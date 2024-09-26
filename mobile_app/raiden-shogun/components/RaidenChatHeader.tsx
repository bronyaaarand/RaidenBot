// components/ChatHeader.tsx

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

type ChatHeaderProps = {
  title: string;
};

const ChatHeader: React.FC<ChatHeaderProps> = ({ title }) => {
  return (
    <View style={styles.headerContainer}>
      <Text style={styles.headerText}>{title}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  headerContainer: {
    padding: 15,
    backgroundColor: '#6200ee', 
  },
  headerText: {
    color: 'white', 
    fontSize: 20,
    fontWeight: 'bold',
  },
});

export default ChatHeader;
