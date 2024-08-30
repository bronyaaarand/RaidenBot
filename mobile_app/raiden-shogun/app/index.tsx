// app/index.tsx

import React from 'react';
import { View, Button, Image, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';

const HomeScreen = () => {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <View style={styles.imageContainer}>
        <Image
          source={{
            uri: 'https://i.pinimg.com/originals/55/a0/c5/55a0c52c088605da8fb9697fd653026a.png',
          }}
          style={styles.image}
          resizeMode="cover"
        />
      </View>

      <View style={styles.navContainer}>
        <Button
          color={"#6200ee"}
          title="Chat with Agent"
          onPress={() => router.push('/(tabs)/bot')}
        />
        <Button
          color={"#6200ee"}
          title="Chat with Customer"
          onPress={() => router.push('/(tabs)/person')}
        />
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
  navItem: {
    backgroundColor: '#6200ee' 
  },
});

export default HomeScreen;
