import React from 'react';
import Layout from '@theme/Layout';
import AuthPage from '../../components/auth/AuthPage';

export default function Auth() {
  return (
    <Layout title="Authentication" description="Authentication page for Physical AI & Humanoid Robotics">
      <AuthPage mode="signin" />
    </Layout>
  );
}