// app/index.tsx

import React, { useEffect, useState } from 'react';
import { View, Text, Image, StyleSheet, FlatList, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';

interface User {
  id: string;
  name: string;
  avatar: string;
}

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

const HomeScreen = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [accessToken, setAccessToken] = useState<string | null>(null); 
  const router = useRouter();

  const fetchUserIds = async (token: string) => {
    try {
      const response = await fetch('https://openapi.zalo.me/v3.0/oa/user/getlist?data={"offset":0,"count":50,"last_interaction_period":"L30D"}', {
        headers: {
          'access_token': token,
        },
      });
      const data = await response.json();
      return data?.data?.users || [];
    } catch (error) {
      console.error('Error fetching user IDs:', error);
      return [];
    }
  };

  const fetchUserDetail = async (user_id: string, token: string): Promise<User | null> => {
    try {
      const response = await fetch(`https://openapi.zalo.me/v3.0/oa/user/detail?data={"user_id":"${user_id}"}`, {
        headers: {
          'access_token': token,
        },
      });
      const data = await response.json();
      return {
        id: user_id,
        name: data.data.display_name,
        avatar: data.data.avatars["240"],
      };
    } catch (error) {
      console.error(`Error fetching details for user ${user_id}:`, error);
      return null;
    }
  };

  const fetchUsers = async () => {
    if (!accessToken) return; 
    const userIds = await fetchUserIds(accessToken);
    const userDetails: (User | null)[] = await Promise.all(userIds.map((user: { user_id: string }) => fetchUserDetail(user.user_id, accessToken)));
    
    setUsers(userDetails.filter((user): user is User => user !== null));
  };

  useEffect(() => {
    const getTokenAndFetchUsers = async () => {
      const token = await getAccessToken();
      if (token) {
        setAccessToken(token); 
      }
    };
    getTokenAndFetchUsers();
  }, []);

  useEffect(() => {
    if (accessToken) {
      fetchUsers();
    }
  }, [accessToken]);

  const renderUser = ({ item }: { item: User }) => (
    <TouchableOpacity onPress={() => router.push({ pathname: './(tabs)/button', params: { userImage: item.avatar, userNavId: item.id, userName: item.name } })}>
      <View style={styles.userContainer}>
        <Image source={{ uri: item.avatar }} style={styles.userAvatar} />
        <Text style={styles.userName}>{item.name}</Text>
      </View>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Danh sách người dùng</Text>
      {users.length > 0 ? (
        <FlatList
          data={users}
          renderItem={renderUser}
          keyExtractor={item => item.id}
        />
      ) : (
        <Text>Đang tải...</Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: 'white',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  userContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  userAvatar: {
    width: 50,
    height: 50,
    borderRadius: 25,
    marginRight: 10,
  },
  userName: {
    fontSize: 16,
  },
});

export default HomeScreen;
