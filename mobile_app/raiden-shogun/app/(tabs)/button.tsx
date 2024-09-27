// app/tabs/button.tsx

import React from 'react';
import { View, TouchableOpacity, Image, StyleSheet, Text } from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';

const ButtonScreen = () => {
  const router = useRouter();
  const params = useLocalSearchParams();
  const userId = params.userNavId as string;
  const userImage = params.userImage as string;
  const userName = params.userName as string;

  return (
    <View style={styles.container}>
      <View style={styles.imageContainer}>
        <Image
          source={{ uri: userImage }}
          style={styles.image}
          resizeMode="cover"
        />
      </View>

      <View style={styles.navContainer}>
        <TouchableOpacity
          style={styles.button}
          onPress={() => router.push(`/(tabs)/bot?user_id=${userId}`)}
        >
          <Text style={styles.buttonText}>AI</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.button}
          onPress={() => router.push(`/(tabs)/person?user_id=${userId}&user_name=${userName}`)}
        >
          <Text style={styles.buttonText}>Khách hàng</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'white',
  },
  imageContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  image: {
    width: 300,
    height: 300,
  },
  navContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
    padding: 30,
  },
  button: {
    flex: 1,
    marginHorizontal: 10,
    padding: 15,
    backgroundColor: '#6200ee',
    borderRadius: 5,
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    textAlign: 'center',
  },
});

export default ButtonScreen;
