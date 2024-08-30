// app/_layout.tsx

import React from 'react';
import { Stack } from 'expo-router';

export default function Layout() {
  return (
    <Stack>
      <Stack.Screen name="(tabs)/bot" options={{ title: 'Chat with Agent' }} />
      <Stack.Screen name="(tabs)/person" options={{ title: 'Chat with Customer' }} />
    </Stack>
  );
}
