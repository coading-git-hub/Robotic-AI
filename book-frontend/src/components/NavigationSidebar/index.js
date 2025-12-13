import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import { useLocation } from '@docusaurus/router';
import { usePluginData } from '@docusaurus/useGlobalData';
import { ThemeClassNames } from '@docusaurus/theme-common';
import styles from './styles.module.css';

/**
 * Custom Navigation Sidebar Component for Physical AI & Humanoid Robotics Course
 * This component provides structured navigation for the 13-week course
 */
export default function NavigationSidebar({ sidebar, path }) {
  const location = useLocation();

  return (
    <nav
      aria-label="Main menu"
      className={clsx(
        ThemeClassNames.docs.docSidebarMenu,
        styles.sidebarMenu,
        'menu',
      )}>
      <ul className={clsx(styles.menuList, 'menu__list')}>
        {sidebar.map((item) => (
          <SidebarItem
            key={item.href || item.docId || item.id}
            item={item}
            isActive={location.pathname === item.href}
            isChild={!item.type || item.type === 'category'}
          />
        ))}
      </ul>
    </nav>
  );
}

function SidebarItem({ item, isActive, isChild }) {
  switch (item.type) {
    case 'link':
      return (
        <li className="menu__list-item">
          <Link
            className={clsx('menu__link', {
              'menu__link--active': isActive,
            })}
            to={item.href}
            {...(item.target && { target: item.target })}
            {...(item.rel && { rel: item.rel })}>
            {item.title}
          </Link>
        </li>
      );
    case 'category':
      return (
        <li className={clsx('menu__list-item', styles.category)}>
          <div className={styles.categoryHeader}>
            <h4 className={styles.categoryTitle}>{item.label}</h4>
          </div>
          <ul className="menu__list">
            {item.items.map((childItem) => (
              <SidebarItem
                key={childItem.href || childItem.docId || childItem.id}
                item={childItem}
                isActive={isActive}
              />
            ))}
          </ul>
        </li>
      );
    default:
      return null;
  }
}