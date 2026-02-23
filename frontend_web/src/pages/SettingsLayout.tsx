import { NavLink, Outlet } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { X } from 'lucide-react';
import { Button, BUTTON_VARIANTS } from '@/components/ui/button';

const navItems = [{ label: 'Connected sources', to: '/settings/sources' }];

export const SettingsLayout = () => {
  return (
    <div className="flex flex-1 h-full justify-center">
      {/* Side navigation within settings */}
      <aside className="w-56 p-4 space-y-2 flex flex-col gap-2 items-start">
        <Button variant="ghost" asChild>
          <NavLink key="go-back" to="/">
            <X />
          </NavLink>
        </Button>
        {navItems.map(item => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              cn(BUTTON_VARIANTS({ variant: isActive ? 'primary' : 'ghost' }))
            }
          >
            {item.label}
          </NavLink>
        ))}
      </aside>

      {/* Active tab content */}
      <main className="w-full max-w-3xl overflow-y-auto px-6">
        <Outlet />
      </main>
    </div>
  );
};
