import React from 'react';
import { View, Text, Image, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';

const HomeScreen = () => {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <View style={styles.imageContainer}>
        <Image
          source={{
            uri: 'https://icones.pro/wp-content/uploads/2022/10/icone-robot-violet.png',
          }}
          style={styles.image}
          resizeMode="cover"
        />
      </View>

      <View style={styles.navContainer}>
        <TouchableOpacity
          style={styles.navItem}
          onPress={() => router.push('/(tabs)/bot')}
        >
          <Text style={styles.navText}>AI</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.navItem}
          onPress={() => router.push('/(tabs)/person')}
        >
          <Text style={styles.navText}>Khách hàng</Text>
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
  navItem: {
    flex: 1,
    backgroundColor: '#6200ee',
    marginHorizontal: 10,
    paddingVertical: 20, 
    borderRadius: 10, 
    justifyContent: 'center',
    alignItems: 'center',
  },
  navText: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
  },
});

export default HomeScreen;
