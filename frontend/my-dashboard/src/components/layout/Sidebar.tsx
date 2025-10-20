import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Box,
  Divider,
  Avatar,
  Chip,
} from '@mui/material';
import {
  Dashboard,
  Timeline,
  Analytics,
  ModelTraining,
  Settings,
  Speed,
  TrendingUp,
  Person,
  ExitToApp,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

// Hooks
import { useAuth } from '../../hooks/useAuth';

const DRAWER_WIDTH = 280;

interface NavItem {
  text: string;
  icon: React.ReactElement;
  path: string;
  badge?: string;
}

const navItems: NavItem[] = [
  {
    text: 'Dashboard',
    icon: <Dashboard />,
    path: '/dashboard',
  },
  {
    text: 'Sessions',
    icon: <Timeline />,
    path: '/sessions',
  },
  {
    text: 'Analytics',
    icon: <Analytics />,
    path: '/analytics',
    badge: 'New',
  },
  {
    text: 'Training',
    icon: <ModelTraining />,
    path: '/training',
  },
  {
    text: 'Settings',
    icon: <Settings />,
    path: '/settings',
  },
];

const Sidebar: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const handleNavigation = (path: string) => {
    navigate(path);
  };

  const handleLogout = () => {
    logout();
  };

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: DRAWER_WIDTH,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: DRAWER_WIDTH,
          boxSizing: 'border-box',
          backgroundColor: 'background.paper',
          borderRight: '1px solid #333',
        },
      }}
    >
      {/* Logo and Brand */}
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <Speed sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
          <Typography variant="h5" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
            LapXcel
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Sim Racing Analytics
          </Typography>
        </motion.div>
      </Box>

      <Divider sx={{ borderColor: '#333' }} />

      {/* User Profile Section */}
      <Box sx={{ p: 2 }}>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            p: 2,
            borderRadius: 2,
            backgroundColor: 'rgba(0, 255, 136, 0.05)',
            border: '1px solid rgba(0, 255, 136, 0.2)',
          }}
        >
          <Avatar
            sx={{
              bgcolor: 'primary.main',
              color: 'black',
              width: 40,
              height: 40,
              mr: 2,
            }}
          >
            {user?.username?.charAt(0).toUpperCase() || 'U'}
          </Avatar>
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600 }} noWrap>
              {user?.username || 'User'}
            </Typography>
            <Typography variant="caption" color="text.secondary" noWrap>
              {user?.email || 'user@example.com'}
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Navigation Items */}
      <List sx={{ flex: 1, px: 1 }}>
        {navItems.map((item, index) => {
          const isActive = location.pathname === item.path;
          
          return (
            <motion.div
              key={item.text}
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: index * 0.1 }}
            >
              <ListItem disablePadding sx={{ mb: 0.5 }}>
                <ListItemButton
                  onClick={() => handleNavigation(item.path)}
                  sx={{
                    borderRadius: 2,
                    mx: 1,
                    backgroundColor: isActive ? 'primary.main' : 'transparent',
                    color: isActive ? 'black' : 'text.primary',
                    '&:hover': {
                      backgroundColor: isActive 
                        ? 'primary.dark' 
                        : 'rgba(255, 255, 255, 0.05)',
                    },
                    transition: 'all 0.2s ease-in-out',
                  }}
                >
                  <ListItemIcon
                    sx={{
                      color: isActive ? 'black' : 'text.secondary',
                      minWidth: 40,
                    }}
                  >
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText
                    primary={item.text}
                    primaryTypographyProps={{
                      fontWeight: isActive ? 600 : 400,
                    }}
                  />
                  {item.badge && (
                    <Chip
                      label={item.badge}
                      size="small"
                      sx={{
                        height: 20,
                        fontSize: '0.7rem',
                        backgroundColor: 'secondary.main',
                        color: 'white',
                      }}
                    />
                  )}
                </ListItemButton>
              </ListItem>
            </motion.div>
          );
        })}
      </List>

      <Divider sx={{ borderColor: '#333' }} />

      {/* Quick Stats */}
      <Box sx={{ p: 2 }}>
        <Typography variant="subtitle2" sx={{ mb: 2, color: 'text.secondary' }}>
          Quick Stats
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="caption" color="text.secondary">
              Best Lap
            </Typography>
            <Typography variant="caption" sx={{ color: 'primary.main', fontWeight: 600 }}>
              1:23.456
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="caption" color="text.secondary">
              Sessions
            </Typography>
            <Typography variant="caption" sx={{ color: 'secondary.main', fontWeight: 600 }}>
              42
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="caption" color="text.secondary">
              Improvement
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <TrendingUp sx={{ fontSize: 12, color: 'success.main', mr: 0.5 }} />
              <Typography variant="caption" sx={{ color: 'success.main', fontWeight: 600 }}>
                +2.1s
              </Typography>
            </Box>
          </Box>
        </Box>
      </Box>

      {/* Logout Button */}
      <Box sx={{ p: 2 }}>
        <ListItemButton
          onClick={handleLogout}
          sx={{
            borderRadius: 2,
            color: 'text.secondary',
            '&:hover': {
              backgroundColor: 'rgba(255, 107, 53, 0.1)',
              color: 'secondary.main',
            },
            transition: 'all 0.2s ease-in-out',
          }}
        >
          <ListItemIcon
            sx={{
              color: 'inherit',
              minWidth: 40,
            }}
          >
            <ExitToApp />
          </ListItemIcon>
          <ListItemText
            primary="Logout"
            primaryTypographyProps={{
              fontSize: '0.875rem',
            }}
          />
        </ListItemButton>
      </Box>
    </Drawer>
  );
};

export default Sidebar;